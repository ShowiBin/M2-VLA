import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
OPENVLA_IMAGE_SIZE = 224

def get_openvla_prompt(instruction: str, tokenized_action: str=None) -> str:
    return f'In: What action should the robot take to {instruction.lower()}?\nOut:'

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

class DualSystemCalvinEvaluation(CalvinBaseModel):

    def __init__(self, model, proprio_projector, noisy_action_projector, action_head, processor, use_x0_prediction=False):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.processor = processor
        self.OFT = model
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.action_head = action_head
        self.use_x0_prediction = use_x0_prediction
        if self.action_head is not None and hasattr(self.action_head, 'use_x0_prediction'):
            self.action_head.use_x0_prediction = use_x0_prediction
        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.action = None
        self.hidden_states = None
        self.obs_buffer = None
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]
        self.depth_max = 6.2
        self.depth_min = 3.5
        self.gripper_depth_max = 2.0
        self.gripper_depth_min = 0
        self.hist_action = []

    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.obs_buffer = None
        self.hist_action = []

    def step(self, obs, instruction, step):
        processed_images = []
        image = obs['rgb_obs']['rgb_static']
        gripper_image = obs['rgb_obs']['rgb_gripper']
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image_resize = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(image_resize).convert('RGB')
        pil_image = center_crop_image(pil_image)
        check_image_format(gripper_image)
        if gripper_image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            gripper_image = resize_image_for_policy(gripper_image, OPENVLA_IMAGE_SIZE)
        gripper_pil_image = Image.fromarray(gripper_image).convert('RGB')
        gripper_pil_image = center_crop_image(gripper_pil_image)
        processed_images.append(pil_image)
        processed_images.append(gripper_pil_image)
        primary_image = processed_images.pop(0)
        prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {instruction.lower()}?<|im_end|>\n<|im_start|>assistant\n'
        inputs = self.processor(prompt, primary_image).to(self.OFT.device, dtype=torch.bfloat16)
        all_wrist_inputs = [self.processor(prompt, processed_images).to(self.OFT.device, dtype=torch.bfloat16)]
        primary_pixel_values = inputs['pixel_values']
        all_wrist_pixel_values = [wrist_inputs['pixel_values'] for wrist_inputs in all_wrist_inputs]
        inputs['pixel_values'] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)
        proprio_state = np.concatenate([obs['robot_obs'][:7], obs['robot_obs'][-1:]])
        proprio_norm_stats = self.OFT.norm_stats['calvin_abc_rlds']['proprio']
        obs['state'] = normalize_proprio(proprio_state, proprio_norm_stats)
        proprio_state = obs['state']
        with torch.no_grad():
            (action, _) = self.OFT.predict_action(**inputs, unnorm_key='calvin_abc_rlds', do_sample=False, proprio=proprio_state, proprio_projector=self.proprio_projector, action_head=self.action_head, noisy_action_projector=self.noisy_action_projector, use_film=False)
        action[:, -1] = 1 - action[:, -1]
        return [action[i] for i in range(min(len(action), 8))]
