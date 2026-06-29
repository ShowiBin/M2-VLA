from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union
import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
overwatch = initialize_overwatch(__name__)
IGNORE_INDEX = -100

class PrismaticVLM(VLM):

    def __init__(self, model_id: str, vision_backbone: VisionBackbone, llm_backbone: LLMBackbone, enable_mixed_precision_training: bool=True, arch_specifier: str='gelu-mlp', **kwargs) -> None:
        super().__init__('prismatic', model_id, vision_backbone, llm_backbone, enable_mixed_precision_training=enable_mixed_precision_training)
        torch.manual_seed(vision_backbone.embed_dim)
        self.arch_specifier = arch_specifier
        if arch_specifier == 'linear':
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith('fused-gelu-mlp'):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith('gelu-mlp'):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f'PrismaticVLM with `arch_specifier = {arch_specifier!r}` is not supported!')
        self.vision_backbone_requires_grad = False
        self.all_module_keys = ['vision_backbone', 'llm_backbone', 'projector']
        self.trainable_module_keys = []
        self.string2idx = {}
        for trigger_string in ['True', 'False', 'Yes', 'No'] + [chr(ord('A') + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(cls, pretrained_checkpoint: Path, model_id: str, vision_backbone: VisionBackbone, llm_backbone: LLMBackbone, enable_mixed_precision_training: bool=True, arch_specifier: str='gelu-mlp', freeze_weights: bool=True, **kwargs) -> PrismaticVLM:
        vlm = cls(model_id, vision_backbone, llm_backbone, enable_mixed_precision_training=enable_mixed_precision_training, arch_specifier=arch_specifier, **kwargs)
        model_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')['model']
        assert 'projector' in model_state_dict and 'llm_backbone' in model_state_dict, 'PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!'
        vlm.projector.load_state_dict(model_state_dict['projector'])
        vlm.llm_backbone.load_state_dict(model_state_dict['llm_backbone'])
        if 'vision_backbone' in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict['vision_backbone'])
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()
        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str]=None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        if stage == 'align':
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)
            self.trainable_module_keys = ['projector']
            self.vision_backbone_requires_grad = False
            overwatch.info(f'[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`', ctx_level=1)
        elif stage in {'finetune', 'vla-train'}:
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.trainable_module_keys = ['projector', 'llm_backbone']
            self.vision_backbone_requires_grad = False
            overwatch.info(f'[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`', ctx_level=1)
        elif stage in {'full-finetune', 'vla-full-train'}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.trainable_module_keys = ['vision_backbone', 'projector', 'llm_backbone']
            self.vision_backbone_requires_grad = True
            overwatch.info(f'[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`', ctx_level=1)
        elif stage in {'last-layer-finetune', 'vla-last-layer-train'}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)
            self.trainable_module_keys = ['llm_backbone']
            self.vision_backbone_requires_grad = False
            overwatch.info(f'[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`', ctx_level=1)
        elif stage in {'vla-sandwich-train'}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)
            self.trainable_module_keys = ['vision_backbone', 'projector', 'llm_backbone']
            self.vision_backbone_requires_grad = True
            overwatch.info(f'[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`', ctx_level=1)
            overwatch.info(f'[TRAINABLE]                 🔥   =>> Projector `{self.arch_specifier}`', ctx_level=1)
        else:
            raise ValueError(f'Stage `{stage}` is not supported for LLaVa! Try < align | finetune >')
        overwatch.debug('##################################################')
        overwatch.debug('#####      Trainable Network Parameters:     #####')
        overwatch.debug('##################################################')
        for (name, param) in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path]=None) -> None:
        assert stage in {'align', 'finetune', 'full-finetune'}, f'Stage {stage} is not supported!'
        if self.arch_specifier.startswith('no-align'):
            overwatch.info(f'PrismaticVLM with `self.arch_specifier = {self.arch_specifier!r}` does not require pretrained weights!', ctx_level=1)
            return
        if stage == 'align':
            overwatch.info('Stage `align` does not require pretrained weights =>> Starting Training', ctx_level=1)
            return
        overwatch.info('Stage `finetune` requires `align` pretrained weights', ctx_level=1)
        if pretrained_checkpoint is not None:
            overwatch.info(f'Loading from Provided Checkpoint `{pretrained_checkpoint}`', ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)['model']
            self.projector.load_state_dict(model_state_dict['projector'])
            return
        (model, scale, _, seed) = run_dir.name.split('+')
        align_dirs = [d for d in run_dir.parent.iterdir() if d.name.startswith(f'{model}+{scale}') and d.name.endswith(f'+stage-align+{seed}')]
        assert len(align_dirs) == 1, 'Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!'
        if (pretrained_checkpoint := (align_dirs[0] / 'checkpoints' / 'latest-checkpoint.pt')).exists():
            overwatch.info(f'Loading from Discovered Checkpoint `{pretrained_checkpoint}`', ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)['model']
            self.projector.load_state_dict(model_state_dict['projector'])
        else:
            raise ValueError(f'Could not find valid `align` checkpoint at {pretrained_checkpoint}!')

    def get_fsdp_wrapping_policy(self) -> Callable:
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()
        prismatic_fsdp_wrapping_policy = partial(_module_wrap_policy, module_classes={LinearProjector, MLPProjector, FusedMLPProjector})
        return partial(_or_policy, policies=[vision_fsdp_wrapping_policy, llm_fsdp_wrapping_policy, prismatic_fsdp_wrapping_policy])

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, multimodal_indices: Optional[torch.LongTensor]=None) -> CausalLMOutputWithPast:
        if input_ids.shape[1] == 1 and past_key_values is not None:
            output = self.llm_backbone(input_ids=input_ids, attention_mask=None, position_ids=None, past_key_values=past_key_values, inputs_embeds=None, labels=None, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            return output
        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError('Invalid `forward()` call!')
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=None, past_key_values=past_key_values, inputs_embeds=None, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full((projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        multimodal_embeddings = torch.cat([input_embeddings[multimodal_indices, :1, :], projected_patch_embeddings, input_embeddings[multimodal_indices, 1:, :]], dim=1)
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat([attention_mask[multimodal_indices, :1], projected_patch_attention_mask, attention_mask[multimodal_indices, 1:]], dim=1)
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full((projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
            multimodal_labels = torch.cat([labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1)
        unimodal_indices = torch.tensor([idx for idx in range(len(input_ids)) if idx not in multimodal_indices], dtype=torch.long, device=multimodal_indices.device)
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels
        else:
            unimodal_embeddings_pad = torch.zeros((len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]), dtype=input_embeddings.dtype, device=input_embeddings.device)
            unimodal_attention_pad = torch.full((len(unimodal_indices), projected_patch_embeddings.shape[1]), False, dtype=attention_mask.dtype, device=attention_mask.device)
            unimodal_labels_pad = torch.full((len(unimodal_indices), projected_patch_embeddings.shape[1]), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])
        return self.llm_backbone(input_ids=None, attention_mask=fused_attention_mask, position_ids=None, past_key_values=past_key_values, inputs_embeds=fused_embeddings, labels=fused_labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

    def prepare_inputs_for_generation(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if past_key_values:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'attention_mask': attention_mask, 'pixel_values': pixel_values, 'past_key_values': past_key_values, 'use_cache': use_cache})
        return model_inputs

    @torch.no_grad()
    def generate_batch(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]], texts: List[str], return_string_probabilities: Optional[List[str]]=None, **kwargs: str) -> Union[List[str], List[List[float]]]:
        tokenizer = self.llm_backbone.tokenizer
        batch_input_ids = [tokenizer(text, truncation=True, return_tensors='pt').input_ids.to(self.device) for text in texts]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for (k, v) in pixel_values.items()}
        else:
            raise ValueError(f'Unsupported `pixel_values` type = {type(pixel_values)}')
        (gen_texts, gen_probabilities) = ([], [])
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast('cuda', dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for (idx, input_ids) in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f'Unsupported `pixel_values` type = {type(pixel_values)}')
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1]:]
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                else:
                    full_out_dict = super().generate(input_ids=input_ids, pixel_values=pixel_values, output_scores=True, return_dict_in_generate=True, **kwargs)
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1]:]
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())
        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.no_grad()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        (image_transform, tokenizer) = (self.vision_backbone.image_transform, self.llm_backbone.tokenizer)
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors='pt').input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for (k, v) in pixel_values.items()}
        else:
            raise ValueError(f'Unsupported `pixel_values` type = {type(pixel_values)}')
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast('cuda', dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            generated_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        return generated_text
