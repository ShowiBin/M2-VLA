from typing import Dict, List, Optional, Union
import numpy as np
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
overwatch = initialize_overwatch(__name__)

class OpenVLA(PrismaticVLM):

    def __init__(self, *args, norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]], action_tokenizer: ActionTokenizer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.no_grad()
    def predict_action(self, image: Union[Img, List[Img]], instruction: str, unnorm_key: Optional[str]=None, **kwargs: str) -> np.ndarray:
        (image_transform, tokenizer) = (self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer)
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role='human', message=f'What action should the robot take to {instruction.lower()}?')
        prompt_text = prompt_builder.get_prompt()
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors='pt').input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            pass
        else:
            raise ValueError(f'Unsupported `tokenizer` type = {type(tokenizer)}')
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for (k, v) in pixel_values.items()}
        else:
            raise ValueError(f'Unsupported `pixel_values` type = {type(pixel_values)}')
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast('cuda', dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            generated_ids = super(PrismaticVLM, self).generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get('mask', np.ones_like(action_norm_stats['q01'], dtype=bool))
        (action_high, action_low) = (np.array(action_norm_stats['q99']), np.array(action_norm_stats['q01']))
        actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, f'Your model was trained on more than one dataset, please pass a `unnorm_key` from the following options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}'
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, f'The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}'
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str]=None) -> int:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]['action']['q01'])

    def get_action_stats(self, unnorm_key: Optional[str]=None) -> Dict:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]['action']
