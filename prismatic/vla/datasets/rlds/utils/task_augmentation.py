from typing import Dict
import tensorflow as tf
from prismatic.vla.datasets.rlds.utils.data_utils import to_padding

def delete_task_conditioning(traj: Dict, keep_image_prob: float) -> Dict:
    if 'language_instruction' not in traj['task']:
        return traj
    image_keys = {key for key in traj['task'].keys() if key.startswith('image_') or key.startswith('depth_')}
    if not image_keys:
        return traj
    traj_len = tf.shape(traj['action'])[0]
    should_keep_images = tf.random.uniform([traj_len]) < keep_image_prob
    should_keep_images |= ~traj['task']['pad_mask_dict']['language_instruction']
    for key in image_keys | {'language_instruction'}:
        should_keep = should_keep_images if key in image_keys else ~should_keep_images
        traj['task'][key] = tf.where(should_keep, traj['task'][key], to_padding(traj['task'][key]))
        traj['task']['pad_mask_dict'][key] = tf.where(should_keep, traj['task']['pad_mask_dict'][key], tf.zeros_like(traj['task']['pad_mask_dict'][key]))
    traj['task']['timestep'] = tf.where(should_keep_images, traj['task']['timestep'], traj_len - 1)
    return traj
