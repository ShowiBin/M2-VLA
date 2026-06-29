from typing import Any, Dict
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg

def rmat_to_euler(rot_mat):
    return tfg.euler.from_rotation_matrix(rot_mat)

def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)

def invert_rmat(rot_mat):
    return tfg.rotation_matrix_3d.inverse(rot_mat)

def rotmat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    (r6_0, r6_1) = (r6[..., 0, :], r6[..., 1, :])
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat

def velocity_act_to_wrist_frame(velocity, wrist_in_robot_frame):
    R_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]
    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)

def rand_swap_exterior_images(img1, img2):
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda : (img1, img2), lambda : (img2, img1))

def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    dt = trajectory['action_dict']['cartesian_velocity'][:, :3]
    dR = trajectory['action_dict']['cartesian_velocity'][:, 3:6]
    trajectory['action'] = tf.concat((dt, dR, 1 - trajectory['action_dict']['gripper_position']), axis=-1)
    (trajectory['observation']['exterior_image_1_left'], trajectory['observation']['exterior_image_2_left']) = rand_swap_exterior_images(trajectory['observation']['exterior_image_1_left'], trajectory['observation']['exterior_image_2_left'])
    trajectory['observation']['proprio'] = tf.concat((trajectory['observation']['cartesian_position'], trajectory['observation']['gripper_position']), axis=-1)
    return trajectory

def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    wrist_act = velocity_act_to_wrist_frame(trajectory['action_dict']['cartesian_velocity'], trajectory['observation']['cartesian_position'])
    trajectory['action'] = tf.concat((wrist_act, trajectory['action_dict']['gripper_position']), axis=-1)
    (trajectory['observation']['exterior_image_1_left'], trajectory['observation']['exterior_image_2_left']) = rand_swap_exterior_images(trajectory['observation']['exterior_image_1_left'], trajectory['observation']['exterior_image_2_left'])
    trajectory['observation']['proprio'] = tf.concat((trajectory['observation']['cartesian_position'], trajectory['observation']['gripper_position']), axis=-1)
    return trajectory

def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    dt = trajectory['action_dict']['cartesian_velocity'][:, :3]
    dR = trajectory['action_dict']['cartesian_velocity'][:, 3:6]
    trajectory['action'] = tf.concat((dt, dR, 1 - trajectory['action_dict']['gripper_position']), axis=-1)
    trajectory['observation']['proprio'] = tf.concat((trajectory['observation']['cartesian_position'], trajectory['observation']['gripper_position']), axis=-1)
    return trajectory

def zero_action_filter(traj: Dict) -> bool:
    DROID_Q01 = tf.convert_to_tensor([-0.7776297926902771, -0.5803514122962952, -0.5795090794563293, -0.6464047729969025, -0.7041108310222626, -0.8895104378461838])
    DROID_Q99 = tf.convert_to_tensor([0.7597932070493698, 0.5726242214441299, 0.7351000607013702, 0.6705610305070877, 0.6464948207139969, 0.8897542208433151])
    DROID_NORM_0_ACT = 2 * (tf.zeros_like(traj['action'][:, :6]) - DROID_Q01) / (DROID_Q99 - DROID_Q01 + 1e-08) - 1
    return tf.reduce_any(tf.math.abs(traj['action'][:, :6] - DROID_NORM_0_ACT) > 1e-05)
