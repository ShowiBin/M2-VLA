import copy
import inspect
import json
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import obs_transforms, traj_transforms
from prismatic.vla.datasets.rlds.utils import goal_relabeling, task_augmentation
from prismatic.vla.datasets.rlds.utils.data_utils import allocate_threads, get_dataset_statistics, normalize_action_and_proprio, pprint_data_mixture, tree_map
overwatch = initialize_overwatch(__name__)
tf.config.set_visible_devices([], 'GPU')

def make_dataset_from_rlds(name: str, data_dir: str, *, train: bool, standardize_fn: Optional[Callable[[dict], dict]]=None, shuffle: bool=True, image_obs_keys: Dict[str, Optional[str]]={}, depth_obs_keys: Dict[str, Optional[str]]={}, state_obs_keys: List[Optional[str]]=(), language_key: Optional[str]=None, action_proprio_normalization_type: ACTION_PROPRIO_NORMALIZATION_TYPE, dataset_statistics: Optional[Union[dict, str]]=None, absolute_action_mask: Optional[List[bool]]=None, action_normalization_mask: Optional[List[bool]]=None, num_parallel_reads: int=tf.data.AUTOTUNE, num_parallel_calls: int=tf.data.AUTOTUNE) -> Tuple[dl.DLataset, dict]:
    REQUIRED_KEYS = {'observation', 'action'}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        if standardize_fn is not None:
            traj = standardize_fn(traj)
        if not all((k in traj for k in REQUIRED_KEYS)):
            raise ValueError(f'Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. Did you write a `standardize_fn`?')
        traj_len = tf.shape(traj['action'])[0]
        old_obs = traj['observation']
        new_obs = {}
        for (new, old) in image_obs_keys.items():
            if old is None:
                new_obs[f'image_{new}'] = tf.repeat('', traj_len)
            else:
                new_obs[f'image_{new}'] = old_obs[old]
        for (new, old) in depth_obs_keys.items():
            if old is None:
                new_obs[f'depth_{new}'] = tf.repeat('', traj_len)
            else:
                new_obs[f'depth_{new}'] = old_obs[old]
        if state_obs_keys:
            new_obs['proprio'] = tf.concat([tf.zeros((traj_len, 1), dtype=tf.float32) if key is None else tf.cast(old_obs[key], tf.float32) for key in state_obs_keys], axis=1)
        new_obs['timestep'] = tf.range(traj_len)
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(f'Language key {language_key} has dtype {traj[language_key].dtype}, but it must be tf.string.')
            task['language_instruction'] = traj.pop(language_key)
        traj = {'observation': new_obs, 'task': task, 'action': tf.cast(traj['action'], tf.float32), 'dataset_name': tf.repeat(name, traj_len)}
        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj['action'].shape[-1]:
                raise ValueError(f"Length of absolute_action_mask ({len(absolute_action_mask)}) does not match action dimension ({traj['action'].shape[-1]}).")
            traj['absolute_action_mask'] = tf.tile(tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None], [traj_len, 1])
        return traj
    builder = tfds.builder(name, data_dir=data_dir)
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, 'r') as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(builder, split='all', shuffle=False, num_parallel_reads=num_parallel_reads).traj_map(restructure, num_parallel_calls)
        dataset_statistics = get_dataset_statistics(full_dataset, hash_dependencies=(str(builder.info), str(state_obs_keys), inspect.getsource(standardize_fn) if standardize_fn is not None else ''), save_dir=builder.data_dir)
    dataset_statistics = tree_map(np.array, dataset_statistics)
    if action_normalization_mask is not None:
        if len(action_normalization_mask) != dataset_statistics['action']['mean'].shape[-1]:
            raise ValueError(f"Length of skip_normalization_mask ({len(action_normalization_mask)}) does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]}).")
        dataset_statistics['action']['mask'] = np.array(action_normalization_mask)
    split = 'train' if train else 'val'
    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(partial(normalize_action_and_proprio, metadata=dataset_statistics, normalization_type=action_proprio_normalization_type), num_parallel_calls)
    return (dataset, dataset_statistics)

def apply_trajectory_transforms(dataset: dl.DLataset, *, train: bool, goal_relabeling_strategy: Optional[str]=None, goal_relabeling_kwargs: dict={}, window_size: int=1, future_action_window_size: int=0, subsample_length: Optional[int]=None, skip_unlabeled: bool=False, max_action: Optional[float]=None, max_proprio: Optional[float]=None, task_augment_strategy: Optional[str]=None, task_augment_kwargs: dict={}, num_parallel_calls: int=tf.data.AUTOTUNE) -> dl.DLataset:
    if skip_unlabeled:
        if 'language_instruction' not in dataset.element_spec['task']:
            raise ValueError('skip_unlabeled=True but dataset does not have language labels.')
        dataset = dataset.filter(lambda x: tf.math.reduce_any(x['task']['language_instruction'] != ''))
    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x['action']) <= max_action))
    if max_proprio is not None and 'proprio' in dataset.element_spec['observation']:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x['observation']['proprio']) <= max_proprio))
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs), num_parallel_calls)
    if train and task_augment_strategy is not None:
        dataset = dataset.traj_map(partial(getattr(task_augmentation, task_augment_strategy), **task_augment_kwargs), num_parallel_calls)
    dataset = dataset.traj_map(partial(traj_transforms.chunk_act_obs, window_size=window_size, future_action_window_size=future_action_window_size), num_parallel_calls)
    if train and subsample_length is not None:
        dataset = dataset.traj_map(partial(traj_transforms.subsample, subsample_length=subsample_length), num_parallel_calls)
    return dataset

def apply_per_dataset_frame_transforms(dataset: dl.DLataset, chunk_filter_fn: Optional[Callable]=None):
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset

def apply_frame_transforms(dataset: dl.DLataset, *, train: bool, image_augment_kwargs: Union[Dict, Dict[str, Dict]]={}, resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]]={}, depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]]={}, num_parallel_calls: int=tf.data.AUTOTUNE) -> dl.DLataset:

    def apply_obs_transform(fn: Callable[[Dict], Dict], frame: Dict) -> Dict:
        frame['task'] = fn(frame['task'])
        frame['observation'] = dl.vmap(fn)(frame['observation'])
        return frame
    dataset = dataset.frame_map(partial(apply_obs_transform, partial(obs_transforms.decode_and_resize, resize_size=resize_size, depth_resize_size=depth_resize_size)), num_parallel_calls)
    if train:

        def aug(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_transform(aug_fn, frame)
        dataset = dataset.frame_map(aug, num_parallel_calls)
    return dataset

def make_single_dataset(dataset_kwargs: dict, *, train: bool, traj_transform_kwargs: dict={}, frame_transform_kwargs: dict={}) -> dl.DLataset:
    (dataset, dataset_statistics) = make_dataset_from_rlds(**dataset_kwargs, train=train)
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)
    dataset = dataset.with_ram_budget(1)
    return (dataset, dataset_statistics['num_trajectories'], dataset_statistics)

def make_interleaved_dataset(dataset_kwargs_list: List[Dict], sample_weights: Optional[List[float]]=None, *, train: bool, shuffle_buffer_size: int, traj_transform_kwargs: Optional[Dict]=None, frame_transform_kwargs: Optional[Dict]=None, batch_size: Optional[int]=None, balance_weights: bool=False, traj_transform_threads: Optional[int]=None, traj_read_threads: Optional[int]=None, max_trajectories_per_dataset: Optional[int]=None) -> dl.DLataset:
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f'sample_weights must be None or have length {len(dataset_kwargs_list)}.')
    if traj_transform_kwargs is None or frame_transform_kwargs is None:
        raise ValueError('Missing `traj_transform_kwargs` and `frame_transform_kwargs`!')
    (dataset_sizes, all_dataset_statistics) = ([], {})
    for dataset_kwargs in dataset_kwargs_list:
        data_kwargs = copy.deepcopy(dataset_kwargs)
        if 'dataset_frame_transform_kwargs' in data_kwargs:
            data_kwargs.pop('dataset_frame_transform_kwargs')
        (_, dataset_statistics) = make_dataset_from_rlds(**data_kwargs, train=train)
        dataset_sizes.append(dataset_statistics['num_transitions'])
        all_dataset_statistics[dataset_kwargs['name']] = dataset_statistics
    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)
    dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)
    overwatch.info('Threads per Dataset: %s', threads_per_dataset)
    overwatch.info('Reads per Dataset: %s', reads_per_dataset)
    overwatch.info('Constructing datasets...')
    datasets = []
    for (dataset_kwargs, threads, reads) in zip(dataset_kwargs_list, threads_per_dataset, reads_per_dataset):
        dataset_frame_transform_kwargs = dataset_kwargs.pop('dataset_frame_transform_kwargs') if 'dataset_frame_transform_kwargs' in dataset_kwargs else {}
        (dataset, _) = make_dataset_from_rlds(**dataset_kwargs, train=train, num_parallel_calls=threads, num_parallel_reads=reads, dataset_statistics=all_dataset_statistics[dataset_kwargs['name']])
        print(f"xsy: Limiting {dataset_kwargs['name']} to {max_trajectories_per_dataset} trajectories")
        if max_trajectories_per_dataset is not None:
            dataset = dataset.take(max_trajectories_per_dataset)
        dataset = apply_trajectory_transforms(dataset.repeat(), **traj_transform_kwargs, num_parallel_calls=threads, train=train).flatten(num_parallel_calls=threads)
        dataset = apply_per_dataset_frame_transforms(dataset, **dataset_frame_transform_kwargs)
        datasets.append(dataset)
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)
    if not train:
        dataset = dataset.take(shuffle_buffer_size).cache()
    dataset = dataset.shuffle(shuffle_buffer_size)
    overwatch.info('Applying frame transforms on dataset...')
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.with_ram_budget(1)
    dataset.sample_weights = sample_weights
    return (dataset, dataset_len, all_dataset_statistics)
