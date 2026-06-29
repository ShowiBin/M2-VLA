import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CodeGenTokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
IGNORE_INDEX = -100

class AlignDataset(Dataset[Dict[str, torch.Tensor]]):

    def __init__(self, chat_json: Path, image_dir: Path, image_transform: ImageTransform, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        (self.chat_json, self.image_dir) = (chat_json, image_dir)
        (self.image_transform, self.tokenizer) = (image_transform, tokenizer)
        self.dataset_type = 'align'
        self.prompt_template = '{caption}' + self.tokenizer.eos_token
        with open(self.chat_json, 'r') as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        (image_path, conversation) = (Path(self.examples[idx]['image']), self.examples[idx]['conversations'])
        assert len(conversation) == 2 and '<image>' not in conversation[-1]['value'], 'Unexpected text!'
        caption = self.prompt_template.format(caption=conversation[-1]['value'].strip())
        input_ids = self.tokenizer(caption, truncation=True, return_tensors='pt').input_ids[0]
        labels = copy.deepcopy(input_ids)
        labels[0] = IGNORE_INDEX
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert('RGB'))
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        modality_lengths = []
        for example in self.examples:
            is_multimodal = 'image' in example
            n_words = sum([len(turn['value'].replace('<image>', '').split()) for turn in example['conversations']])
            modality_lengths.append((is_multimodal, n_image_patches + n_words if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):

    def __init__(self, instruct_json: Path, image_dir: Path, image_transform: ImageTransform, tokenizer: PreTrainedTokenizerBase, prompt_builder_fn: Type[PromptBuilder]) -> None:
        super().__init__()
        (self.instruct_json, self.image_dir) = (instruct_json, image_dir)
        (self.image_transform, self.tokenizer) = (image_transform, tokenizer)
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = 'finetune'
        with open(self.instruct_json, 'r') as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.examples[idx]['conversations']
        (prompt_builder, input_ids, labels) = (self.prompt_builder_fn(model_family='prismatic'), [], [])
        for (turn_idx, turn) in enumerate(conversation):
            msg = prompt_builder.add_turn(turn['from'], turn['value'])
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()
            elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                pass
            elif isinstance(self.tokenizer, Qwen2TokenizerFast):
                if turn['from'] == 'gpt' and turn_idx == len(conversation) - 1:
                    msg += prompt_builder.eos
            else:
                raise ValueError(f'Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!')
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
            turn_labels = [IGNORE_INDEX for _ in range(len(turn_input_ids))] if turn_idx % 2 == 0 else list(turn_input_ids)
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)
        (input_ids, labels) = (torch.tensor(input_ids), torch.tensor(labels))
        (input_ids, labels) = (input_ids[:self.tokenizer.model_max_length], labels[:self.tokenizer.model_max_length])
        if 'image' in self.examples[idx]:
            image_path = Path(self.examples[idx]['image'])
            labels[0] = IGNORE_INDEX
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert('RGB'))
            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        else:
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        modality_lengths = []
        for example in self.examples:
            is_multimodal = 'image' in example
            n_words = sum([len(turn['value'].split()) for turn in example['conversations']])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
