import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
json_numpy.patch()
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
DATE = time.strftime('%Y_%m_%d')
DATE_TIME = time.strftime('%Y_%m_%d-%H_%M_%S')
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
OPENVLA_IMAGE_SIZE = 224
np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

def model_is_on_hf_hub(model_path: str) -> bool:
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False

def update_auto_map(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return
    config_path = os.path.join(pretrained_checkpoint, 'config.json')
    if not os.path.exists(config_path):
        print(f'Warning: No config.json found at {config_path}')
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(pretrained_checkpoint, f'config.json.back.{timestamp}')
    shutil.copy2(config_path, backup_path)
    print(f'Created backup of original config at: {os.path.abspath(backup_path)}')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['auto_map'] = {'AutoConfig': 'configuration_prismatic.OpenVLAConfig', 'AutoModelForVision2Seq': 'modeling_prismatic.OpenVLAForActionPrediction'}
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Updated config.json at: {os.path.abspath(config_path)}')
    print('Changes made:')
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')

def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    (path1, path2) = (Path(path1), Path(path2))
    if path1.stat().st_size != path2.stat().st_size:
        return False
    return filecmp.cmp(path1, path2, shallow=False)

def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    if os.path.exists(checkpoint_filepath):
        match = check_identical_files(curr_filepath, checkpoint_filepath)
        if not match:
            print(f'\n------------------------------------------------------------------------------------------------\nFound mismatch between:\nCurrent:   {curr_filepath}\nCheckpoint: {checkpoint_filepath}\n')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f'{checkpoint_filepath}.back.{timestamp}'
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f'Created backup of original checkpoint file at: {os.path.abspath(backup_path)}')
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f'Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}')
            print(f'Changes complete. The checkpoint will now use the current version of {file_type}\n------------------------------------------------------------------------------------------------\n')
    else:
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(f'\n------------------------------------------------------------------------------------------------\nNo {file_type} found in checkpoint directory.\nCopied current version from: {curr_filepath}\nTo checkpoint location: {os.path.abspath(checkpoint_filepath)}\n------------------------------------------------------------------------------------------------\n')

def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return
    curr_files = {'modeling_prismatic.py': None, 'configuration_prismatic.py': None}
    for (root, _, files) in os.walk('./prismatic/'):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)
    for (filename, curr_filepath) in curr_files.items():
        if curr_filepath is None:
            print(f'WARNING: `{filename}` is not found anywhere in the current directory.')
            continue
        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)

def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    assert os.path.isdir(pretrained_checkpoint), f'Checkpoint path must be a directory: {pretrained_checkpoint}'
    for filename in (f'{file_pattern}.pt', f'{file_pattern}--checkpoint.pt'):
        checkpoint_path = os.path.join(pretrained_checkpoint, filename)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and 'checkpoint' in filename and '.back.' not in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            if os.path.isfile(full_path):
                checkpoint_files.append(full_path)
    available = ', '.join(sorted(os.listdir(pretrained_checkpoint)))
    assert len(checkpoint_files) == 1, f'Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}. Available files: {available}'
    return checkpoint_files[0]

def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, weights_only=True)
    new_state_dict = {}
    for (k, v) in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_component_state_dict_v1(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, weights_only=True)
    new_state_dict = {}
    for (k, v) in state_dict.items():
        new_state_dict[k] = v
    return new_state_dict

def get_vla(cfg: Any) -> torch.nn.Module:
    print('Instantiating pretrained VLA policy...')
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register('openvla', OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)
    vla = AutoModelForVision2Seq.from_pretrained(cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16, load_in_8bit=cfg.load_in_8bit, load_in_4bit=cfg.load_in_4bit, low_cpu_mem_usage=False, trust_remote_code=False)
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()
    if not cfg.load_in_8bit and (not cfg.load_in_4bit):
        vla = vla.to(DEVICE)
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla

def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, target_modules='all-linear', init_lora_weights='gaussian')
    vla = get_peft_model(vla, lora_config)
    new_vision_backbone = FiLMedPrismaticVisionBackbone(vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim)
    vla.model.vision_backbone = new_vision_backbone
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, 'vision_backbone')
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)
    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)
    return vla

def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    if model_is_on_hf_hub(checkpoint_path):
        dataset_statistics_path = hf_hub_download(repo_id=checkpoint_path, filename='dataset_statistics.json')
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, 'dataset_statistics.json')
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, 'r') as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print('WARNING: No local dataset_statistics.json file found for current checkpoint.\nYou can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`.')

def get_processor(cfg: Any) -> AutoProcessor:
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)

def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=proprio_dim).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {'VLA-Adapter/LIBERO-Spatial-Pro': 'proprio_projector--checkpoint.pt', 'VLA-Adapter/LIBERO-Object-Pro': 'proprio_projector--checkpoint.pt', 'VLA-Adapter/LIBERO-Goal-Pro': 'proprio_projector--checkpoint.pt', 'VLA-Adapter/LIBERO-Long-Pro': 'proprio_projector--checkpoint.pt'}
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError('Unsupported HF Hub pretrained checkpoint found!')
        proprio_projector_path = hf_hub_download(repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint])
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, 'proprio_projector')
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)
    return proprio_projector

def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
    noisy_action_projector = NoisyActionProjector(llm_dim=llm_dim).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, 'noisy_action_projector')
    state_dict = load_component_state_dict(checkpoint_path)
    noisy_action_projector.load_state_dict(state_dict)
    return noisy_action_projector

def get_action_head(cfg: Any, llm_dim: int) -> Union[L1RegressionActionHead]:
    if not hasattr(cfg, 'use_pro_version'):
        if 'Pro' in cfg.pretrained_checkpoint or 'Pro' in cfg.save_version:
            cfg.use_pro_version = True
        else:
            cfg.use_pro_version = False
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, use_pro_version=cfg.use_pro_version)
    else:
        raise ValueError('Either use_l1_regression or use_diffusion must be True')
    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()
    action_head.memory_bank.initialize(key_dim=llm_dim * 3, value_dim=ACTION_DIM * 8, device=DEVICE)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {'VLA-Adapter/LIBERO-Spatial-Pro': 'action_head--checkpoint.pt', 'VLA-Adapter/LIBERO-Object-Pro': 'action_head--checkpoint.pt', 'VLA-Adapter/LIBERO-Goal-Pro': 'action_head--checkpoint.pt', 'VLA-Adapter/LIBERO-Long-Pro': 'action_head--checkpoint.pt'}
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
            raise ValueError('Unsupported HF Hub pretrained checkpoint found!')
        action_head_path = hf_hub_download(repo_id=cfg.pretrained_checkpoint, filename=model_path_to_action_head_name[cfg.pretrained_checkpoint])
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, 'action_head')
        state_dict = load_component_state_dict(checkpoint_path)
    action_head.load_state_dict(state_dict)
    return action_head

def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method='lanczos3', antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()

def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    assert image.shape.ndims in (3, 4), 'Image must be 3D or 4D tensor'
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack([height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths], axis=1)
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE))
    if expanded_dims:
        image = image[0]
    return image

def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    batch_size = 1
    crop_scale = 0.9
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert('RGB')

def check_image_format(image: Any) -> None:
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8
    assert is_numpy_array and has_correct_shape and has_correct_dtype, 'Incorrect image format detected! Make sure that the input image is a numpy array with shape (H, W, 3) and dtype np.uint8!'

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get('mask', np.ones_like(norm_stats['min'], dtype=bool))
        (proprio_high, proprio_low) = (np.array(norm_stats['max']), np.array(norm_stats['min']))
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get('mask', np.ones_like(norm_stats['q01'], dtype=bool))
        (proprio_high, proprio_low) = (np.array(norm_stats['q99']), np.array(norm_stats['q01']))
    else:
        raise ValueError('Unsupported action/proprio normalization type detected!')
    normalized_proprio = np.clip(np.where(mask, 2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-08) - 1, proprio), a_min=-1.0, a_max=1.0)
    return normalized_proprio

def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    processed_images = []
    for image in images:
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(image).convert('RGB')
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)
        processed_images.append(pil_image)
    return processed_images

def get_vla_action(cfg: Any, vla: torch.nn.Module, processor: Any, obs: Dict[str, Any], task_label: str, action_head: Optional[torch.nn.Module]=None, proprio_projector: Optional[torch.nn.Module]=None, noisy_action_projector: Optional[torch.nn.Module]=None, use_film: bool=False, use_minivlm: bool=False) -> List[np.ndarray]:
    with torch.no_grad():
        all_images = [obs['full_image']]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if 'wrist' in k])
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        if not use_minivlm:
            prompt = f'In: What action should the robot take to {task_label.lower()}?\nOut:'
        else:
            prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n<|im_start|>assistant\n'
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images]
            primary_pixel_values = inputs['pixel_values']
            all_wrist_pixel_values = [wrist_inputs['pixel_values'] for wrist_inputs in all_wrist_inputs]
            inputs['pixel_values'] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)
        proprio = None
        if cfg.use_proprio:
            proprio = obs['state']
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]['proprio']
            obs['state'] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs['state']
        if action_head is None:
            (action, _) = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            (action, _) = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=proprio, proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector, action_head=action_head, use_film=use_film)
    return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]

def get_action_from_server(observation: Dict[str, Any], server_endpoint: str='http://0.0.0.0:8777/act') -> Dict[str, Any]:
    response = requests.post(server_endpoint, json=observation)
    return response.json()
