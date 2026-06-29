import math
import os
import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.robot_utils import DATE, DATE_TIME
def get_libero_env(task, model_family, resolution=256):
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env_args = {'bddl_file_name': task_bddl_file, 'camera_heights': resolution, 'camera_widths': resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return (env, task_description)

def get_libero_dummy_action(model_family: str):
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    img = obs['agentview_image']
    img = img[::-1, ::-1]
    return img

def get_libero_wrist_image(obs):
    img = obs['robot0_eye_in_hand_image']
    img = img[::-1, ::-1]
    return img

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, save_version=None):
    rollout_dir = f'./rollouts/{save_version}/{DATE}'
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(' ', '_').replace('\n', '_').replace('.', '_')[:50]
    mp4_path = f'{rollout_dir}/{DATE_TIME}--episode={idx}--task={processed_task_description}.mp4'
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f'Saved rollout MP4 at path {mp4_path}')
    if log_file is not None:
        log_file.write(f'Saved rollout MP4 at path {mp4_path}\n')
    return mp4_path

def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den
