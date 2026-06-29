import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import wandb
sys.path.append('../..')
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, get_libero_wrist_image, quat2axisangle, save_rollout_video
from experiments.robot.openvla_utils import get_action_head, get_noisy_action_projector, get_processor, get_proprio_projector, resize_image_for_policy
from experiments.robot.robot_utils import DATE_TIME, get_action, get_image_resize_size, get_model, invert_gripper_action, normalize_gripper_action, set_seed_everywhere
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = 'libero_spatial'
    LIBERO_OBJECT = 'libero_object'
    LIBERO_GOAL = 'libero_goal'
    LIBERO_10 = 'libero_10'
    LIBERO_90 = 'libero_90'
TASK_MAX_STEPS = {TaskSuite.LIBERO_SPATIAL: 220, TaskSuite.LIBERO_OBJECT: 280, TaskSuite.LIBERO_GOAL: 300, TaskSuite.LIBERO_10: 520, TaskSuite.LIBERO_90: 400}
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

@dataclass
class GenerateConfig:
    model_family: str = 'openvla'
    pretrained_checkpoint: Union[str, Path] = ''
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ''
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = 'DEFAULT'
    env_img_res: int = 256
    run_id_note: Optional[str] = None
    local_log_dir: str = './experiments/logs'
    use_wandb: bool = False
    wandb_entity: str = 'your-wandb-entity'
    wandb_project: str = 'your-wandb-project'
    seed: int = 7
    save_version: str = 'm2-vla'
    use_pro_version: bool = True
    phase: str = 'Evaluation'

def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, 'pretrained_checkpoint must not be None!'
    if 'image_aug' in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, 'Expecting `center_crop==True` because model was trained with image augmentations!'
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), 'Cannot use both 8-bit and 4-bit quantization!'
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f'Invalid task suite: {cfg.task_suite_name}'

def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)
    noisy_action_projector = None
    processor = None
    if cfg.model_family == 'openvla':
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)
    return (model, action_head, proprio_projector, noisy_action_projector, processor)

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f'{unnorm_key}_no_noops' in model.norm_stats:
        unnorm_key = f'{unnorm_key}_no_noops'
    assert unnorm_key in model.norm_stats, f'Action un-norm key {unnorm_key} not found in VLA `norm_stats`!'
    cfg.unnorm_key = unnorm_key

def setup_logging(cfg: GenerateConfig):
    run_id = f'EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}'
    if cfg.run_id_note is not None:
        run_id += f'--{cfg.run_id_note}'
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + '.txt')
    log_file = open(local_log_filepath, 'w')
    logger.info(f'Logging to local log file: {local_log_filepath}')
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return (log_file, local_log_filepath, run_id)

def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != 'DEFAULT':
        with open(cfg.initial_states_path, 'r') as f:
            all_initial_states = json.load(f)
        log_message(f'Using initial states from {cfg.initial_states_path}', log_file)
        return (initial_states, all_initial_states)
    else:
        log_message('Using default initial states', log_file)
        return (initial_states, None)

def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {'full_image': img_resized, 'wrist_image': wrist_img_resized, 'state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))}
    return (observation, img)

def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == 'openvla':
        action = invert_gripper_action(action)
    return action

def run_episode(cfg: GenerateConfig, env, task_description: str, model, resize_size, processor=None, action_head=None, proprio_projector=None, noisy_action_projector=None, initial_state=None, log_file=None):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f'WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK {{NUM_ACTIONS_CHUNK}} constant defined in prismatic.vla.constants. Executing the full action chunk is recommended.')
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                (obs, reward, done, info) = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            (observation, img) = prepare_observation(obs, resize_size)
            replay_images.append(img)
            if len(action_queue) == 0:
                actions = get_action(cfg, model, observation, task_description, processor=processor, action_head=action_head, proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector, use_film=cfg.use_film, use_minivlm=cfg.use_minivlm)
                action_queue.extend(actions)
            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            (obs, reward, done, info) = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        log_message(f'Episode error: {e}', log_file)
    return (success, replay_images)

def run_task(cfg: GenerateConfig, task_suite, task_id: int, model, resize_size, processor=None, action_head=None, proprio_projector=None, noisy_action_projector=None, total_episodes=0, total_successes=0, log_file=None, save_version=None):
    task = task_suite.get_task(task_id)
    (initial_states, all_initial_states) = load_initial_states(cfg, task_suite, task_id, log_file)
    (env, task_description) = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    (task_episodes, task_successes) = (0, 0)
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f'\nTask: {task_description}', log_file)
        if cfg.initial_states_path == 'DEFAULT':
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(' ', '_')
            episode_key = f'demo_{episode_idx}'
            if not all_initial_states[initial_states_task_key][episode_key]['success']:
                log_message(f'Skipping task {task_id} episode {episode_idx} due to failed expert demo!', log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]['initial_state'])
        log_message(f'Starting episode {task_episodes + 1}...', log_file)
        (success, replay_images) = run_episode(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector, initial_state, log_file)
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
        save_rollout_video(replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, save_version=save_version)
        log_message(f'Success: {success}', log_file)
        log_message(f'# episodes completed so far: {total_episodes}', log_file)
        log_message(f'# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)', log_file)
    task_completion_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_completion_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f'Current task completion rate: {task_completion_rate}', log_file)
    log_message(f'Current total completion rate: {total_completion_rate}', log_file)
    env.close()
    del env
    if cfg.use_wandb:
        wandb.log({f'completion_rate/{task_description}': task_completion_rate, f'num_episodes/{task_description}': task_episodes})
    return (total_episodes, total_successes)

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    (model, action_head, proprio_projector, noisy_action_projector, processor) = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    (log_file, local_log_filepath, run_id) = setup_logging(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    log_message(f'Task suite: {cfg.task_suite_name}', log_file)
    (total_episodes, total_successes) = (0, 0)
    for task_id in tqdm.tqdm(range(num_tasks)):
        (total_episodes, total_successes) = run_task(cfg, task_suite, task_id, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector, total_episodes, total_successes, log_file, cfg.save_version)
    final_completion_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message('Final results:', log_file)
    log_message(f'Total episodes: {total_episodes}', log_file)
    log_message(f'Total successes: {total_successes}', log_file)
    log_message(f'Overall completion rate: {final_completion_rate:.4f} ({final_completion_rate * 100:.1f}%)', log_file)
    if cfg.use_wandb:
        wandb.log({'completion_rate/total': final_completion_rate, 'num_episodes/total': total_episodes})
        wandb.save(local_log_filepath)
    if log_file:
        log_file.close()
    return final_completion_rate
if __name__ == '__main__':
    eval_libero()
