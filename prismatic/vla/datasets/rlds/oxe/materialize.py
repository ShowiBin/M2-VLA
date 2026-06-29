from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
overwatch = initialize_overwatch(__name__)

def make_oxe_dataset_kwargs(dataset_name: str, data_root_dir: Path, load_camera_views: Tuple[str]=('primary',), load_depth: bool=False, load_proprio: bool=True, load_language: bool=True, action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE) -> Dict[str, Any]:
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    if dataset_kwargs['action_encoding'] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6, ActionEncoding.JOINT_POS_BIMANUAL]:
        raise ValueError(f'Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 & JOINT_POS_BIMANUAL actions supported!')
    if dataset_kwargs['action_encoding'] is ActionEncoding.EEF_POS:
        dataset_kwargs['absolute_action_mask'] = [False] * 6 + [True]
        dataset_kwargs['action_normalization_mask'] = [True] * 6 + [False]
    elif dataset_kwargs['action_encoding'] is ActionEncoding.EEF_R6:
        dataset_kwargs['absolute_action_mask'] = [False] * 9 + [True]
        dataset_kwargs['action_normalization_mask'] = [True] * 9 + [False]
    elif dataset_kwargs['action_encoding'] is ActionEncoding.JOINT_POS_BIMANUAL:
        dataset_kwargs['absolute_action_mask'] = [True] * 14
        dataset_kwargs['action_normalization_mask'] = [True] * 14
    dataset_kwargs['action_proprio_normalization_type'] = action_proprio_normalization_type
    if len((missing_keys := (set(load_camera_views) - set(dataset_kwargs['image_obs_keys'])))) > 0:
        raise ValueError(f'Cannot load `{dataset_name}`; missing camera views `{missing_keys}`')
    dataset_kwargs['image_obs_keys'] = {k: v for (k, v) in dataset_kwargs['image_obs_keys'].items() if k in load_camera_views}
    dataset_kwargs['depth_obs_keys'] = {k: v for (k, v) in dataset_kwargs['depth_obs_keys'].items() if k in load_camera_views}
    dataset_kwargs.pop('state_encoding')
    dataset_kwargs.pop('action_encoding')
    if not load_depth:
        dataset_kwargs.pop('depth_obs_keys')
    if not load_proprio:
        dataset_kwargs.pop('state_obs_keys')
    if load_language:
        dataset_kwargs['language_key'] = 'language_instruction'
    dataset_kwargs['standardize_fn'] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]
    if 'aux_kwargs' in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop('aux_kwargs'))
    return {'name': dataset_name, 'data_dir': str(data_root_dir), **dataset_kwargs}

def get_oxe_dataset_kwargs_and_weights(data_root_dir: Path, mixture_spec: List[Tuple[str, float]], load_camera_views: Tuple[str]=('primary',), load_depth: bool=False, load_proprio: bool=True, load_language: bool=True, action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE) -> Tuple[Dict[str, Any], List[float]]:
    (included_datasets, filtered_mixture_spec) = (set(), [])
    for (d_name, d_weight) in mixture_spec:
        if d_name in included_datasets:
            overwatch.warning(f'Skipping Duplicate Dataset: `{(d_name, d_weight)}`')
            continue
        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))
    (per_dataset_kwargs, sampling_weights) = ([], [])
    for (d_name, d_weight) in filtered_mixture_spec:
        try:
            per_dataset_kwargs.append(make_oxe_dataset_kwargs(d_name, data_root_dir, load_camera_views, load_depth, load_proprio, load_language, action_proprio_normalization_type))
            sampling_weights.append(d_weight)
        except ValueError as e:
            overwatch.warning(f'Skipping `{d_name}` due to Error: {e}')
    return (per_dataset_kwargs, sampling_weights)
