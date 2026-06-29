from typing import Dict
import tensorflow as tf
from prismatic.vla.datasets.rlds.utils.data_utils import tree_merge

def uniform(traj: Dict) -> Dict:
    traj_len = tf.shape(tf.nest.flatten(traj['observation'])[0])[0]
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj['observation'])
    traj['task'] = tree_merge(traj['task'], goal)
    return traj
