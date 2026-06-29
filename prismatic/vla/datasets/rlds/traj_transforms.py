import logging
from typing import Dict
import tensorflow as tf

def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int=0) -> Dict:
    traj_len = tf.shape(traj['action'])[0]
    action_dim = traj['action'].shape[-1]
    effective_traj_len = traj_len - future_action_window_size
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size])
    action_chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1 + future_action_window_size), [effective_traj_len, window_size + future_action_window_size]) + tf.broadcast_to(tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size + future_action_window_size])
    floored_chunk_indices = tf.maximum(chunk_indices, 0)
    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    traj['observation'] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj['observation'])
    traj['action'] = tf.gather(traj['action'], floored_action_chunk_indices)
    traj['observation']['pad_mask'] = chunk_indices >= 0
    traj['task'] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj['task'])
    traj['dataset_name'] = tf.gather(traj['dataset_name'], tf.range(effective_traj_len))
    traj['absolute_action_mask'] = tf.gather(traj['absolute_action_mask'], tf.range(effective_traj_len))
    return traj

def subsample(traj: Dict, subsample_length: int) -> Dict:
    traj_len = tf.shape(traj['action'])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj

def add_pad_mask_dict(traj: Dict) -> Dict:
    traj_len = tf.shape(traj['action'])[0]
    for key in ['observation', 'task']:
        pad_mask_dict = {}
        for subkey in traj[key]:
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
        traj[key]['pad_mask_dict'] = pad_mask_dict
    return traj
