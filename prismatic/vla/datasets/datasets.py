from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        (dataset_name, current_action) = (rlds_batch['dataset_name'], rlds_batch['action'][0])
        img = Image.fromarray(rlds_batch['observation']['image_primary'][0])
        lang = rlds_batch['task']['language_instruction'].decode().lower()
        actions = rlds_batch['action']
        prompt_builder = self.prompt_builder_fn('openvla')
        future_actions = rlds_batch['action'][1:]
        if self.use_minivlm:
            self.prompt_builder_fn = QwenPromptBuilder
            prompt_builder = self.prompt_builder_fn('openvla')
            future_actions_string = self.action_tokenizer(future_actions, self.use_minivlm)
            current_action_string = self.action_tokenizer(current_action, self.use_minivlm)
            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]
            action_chunk_len = len(flattened_action_chunk_string)
            conversation = [{'from': 'human', 'value': f'What action should the robot take to {lang}?'}, {'from': 'gpt', 'value': ''}]
            for turn in conversation:
                prompt_builder.add_turn(turn['from'], turn['value'])
            prompt = prompt_builder.get_prompt()
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            if len(input_ids) >= 3:
                del input_ids[-3]
                del input_ids[-2]
                del input_ids[-1]
            if NUM_TOKENS < len(flattened_action_chunk_string):
                input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                input_ids = input_ids + flattened_action_chunk_string + extended_array
            labels = list(input_ids)
            action_chunk_len = NUM_TOKENS
        else:
            future_actions_string = ''.join(self.action_tokenizer(future_actions, use_minivlm=False))
            current_action_string = self.action_tokenizer(current_action, use_minivlm=False)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)
            conversation = [{'from': 'human', 'value': f'What action should the robot take to {lang}?'}, {'from': 'gpt', 'value': action_chunk_string[0]}]
            action_chunk_len = 1
            for turn in conversation:
                prompt_builder.add_turn(turn['from'], turn['value'])
            prompt = prompt_builder.get_prompt()
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            labels = list(input_ids)
        (input_ids, labels) = (torch.tensor(input_ids), torch.tensor(labels))
        pixel_values = self.image_transform(img)
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch['observation'].keys():
                if 'wrist' in k:
                    img_wrist = Image.fromarray(rlds_batch['observation'][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict['pixel_values_wrist'] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and 'proprio' in rlds_batch['observation']:
            proprio = rlds_batch['observation']['proprio']
            return_dict['proprio'] = proprio
        return return_dict

class RLDSDataset(IterableDataset):

    def __init__(self, data_root_dir: Path, data_mix: str, batch_transform: RLDSBatchTransform, resize_resolution: Tuple[int, int], shuffle_buffer_size: int=256000, train: bool=True, image_aug: bool=False, max_trajectories_per_dataset: Optional[int]=None) -> None:
        (self.data_root_dir, self.data_mix, self.batch_transform) = (data_root_dir, data_mix, batch_transform)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]
        if 'aloha' in self.data_mix:
            load_camera_views = ('primary', 'left_wrist', 'right_wrist')
        else:
            load_camera_views = ('primary', 'wrist')
        (per_dataset_kwargs, weights) = get_oxe_dataset_kwargs_and_weights(self.data_root_dir, mixture_spec, load_camera_views=load_camera_views, load_depth=False, load_proprio=True, load_language=True, action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE)
        rlds_config = dict(traj_transform_kwargs=dict(window_size=1, future_action_window_size=NUM_ACTIONS_CHUNK - 1, skip_unlabeled=True, goal_relabeling_strategy='uniform'), frame_transform_kwargs=dict(resize_size=resize_resolution, num_parallel_calls=16), dataset_kwargs_list=per_dataset_kwargs, shuffle_buffer_size=shuffle_buffer_size, sample_weights=weights, balance_weights=True, traj_transform_threads=len(mixture_spec), traj_read_threads=len(mixture_spec), train=train, max_trajectories_per_dataset=max_trajectories_per_dataset)
        if image_aug:
            (rlds_config['frame_transform_kwargs'].update({'image_augment_kwargs': dict(random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]), random_brightness=[0.2], random_contrast=[0.8, 1.2], random_saturation=[0.8, 1.2], random_hue=[0.05], augment_order=['random_resized_crop', 'random_brightness', 'random_contrast', 'random_saturation', 'random_hue'])}),)
        (self.dataset, self.dataset_length, self.dataset_statistics) = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError('IterableDataset does not implement map-style __getitem__; see __iter__ instead!')

class EpisodicRLDSDataset(RLDSDataset):

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config['dataset_kwargs_list']
        assert len(per_dataset_kwargs) == 1, 'Only support single-dataset `mixes` for episodic datasets.'
        return make_single_dataset(per_dataset_kwargs[0], train=rlds_config['train'], traj_transform_kwargs=rlds_config['traj_transform_kwargs'], frame_transform_kwargs=rlds_config['frame_transform_kwargs'])

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [self.batch_transform(tree_map(lambda x: x[i], rlds_batch)) for i in range(rlds_batch['action'].shape[0])]
            yield out

class DummyDataset(Dataset):

    def __init__(self, action_tokenizer: ActionTokenizer, base_tokenizer: PreTrainedTokenizerBase, image_transform: ImageTransform, prompt_builder_fn: Type[PromptBuilder]) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_statistics = {'dummy_dataset': {'action': {'q01': np.zeros((7,), dtype=np.float32), 'q99': np.ones((7,), dtype=np.float32)}}}

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = 'do something spectacular'
        prompt_builder = self.prompt_builder_fn('openvla')
        conversation = [{'from': 'human', 'value': f'What action should the robot take to {instruction}?'}, {'from': 'gpt', 'value': self.action_tokenizer(action)}]
        for turn in conversation:
            prompt_builder.add_turn(turn['from'], turn['value'])
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        (input_ids, labels) = (torch.tensor(input_ids), torch.tensor(labels))
        pixel_values = self.image_transform(image)
        labels[:-(len(action) + 1)] = IGNORE_INDEX
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
