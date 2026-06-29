from typing import Optional, Tuple
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.llm import LLaMa2LLMBackbone, LLMBackbone, MistralLLMBackbone, PhiLLMBackbone
from prismatic.models.backbones.llm.qwen25 import Qwen25LLMBackbone
from prismatic.models.backbones.vision import CLIPViTBackbone, DinoCLIPViTBackbone, DinoSigLIPViTBackbone, DinoV2ViTBackbone, ImageTransform, IN1KViTBackbone, SigLIPViTBackbone, VisionBackbone
from prismatic.models.vlms import PrismaticVLM
VISION_BACKBONES = {'clip-vit-l': {'cls': CLIPViTBackbone, 'kwargs': {'default_image_size': 224}}, 'siglip-vit-so400m': {'cls': SigLIPViTBackbone, 'kwargs': {'default_image_size': 224}}, 'dinov2-vit-l': {'cls': DinoV2ViTBackbone, 'kwargs': {'default_image_size': 224}}, 'in1k-vit-l': {'cls': IN1KViTBackbone, 'kwargs': {'default_image_size': 224}}, 'dinosiglip-vit-so-224px': {'cls': DinoSigLIPViTBackbone, 'kwargs': {'default_image_size': 224}}, 'clip-vit-b': {'cls': CLIPViTBackbone, 'kwargs': {'default_image_size': 224}}, 'clip-vit-l-336px': {'cls': CLIPViTBackbone, 'kwargs': {'default_image_size': 336}}, 'siglip-vit-b16-224px': {'cls': SigLIPViTBackbone, 'kwargs': {'default_image_size': 224}}, 'siglip-vit-b16-256px': {'cls': SigLIPViTBackbone, 'kwargs': {'default_image_size': 256}}, 'siglip-vit-b16-384px': {'cls': SigLIPViTBackbone, 'kwargs': {'default_image_size': 384}}, 'siglip-vit-so400m-384px': {'cls': SigLIPViTBackbone, 'kwargs': {'default_image_size': 384}}, 'dinoclip-vit-l-336px': {'cls': DinoCLIPViTBackbone, 'kwargs': {'default_image_size': 336}}, 'dinosiglip-vit-so-384px': {'cls': DinoSigLIPViTBackbone, 'kwargs': {'default_image_size': 384}}}
LLM_BACKBONES = {'llama2-7b-pure': {'cls': LLaMa2LLMBackbone, 'kwargs': {}}, 'llama2-13b-pure': {'cls': LLaMa2LLMBackbone, 'kwargs': {}}, 'llama2-7b-chat': {'cls': LLaMa2LLMBackbone, 'kwargs': {}}, 'llama2-13b-chat': {'cls': LLaMa2LLMBackbone, 'kwargs': {}}, 'vicuna-v15-7b': {'cls': LLaMa2LLMBackbone, 'kwargs': {}}, 'vicuna-v15-13b': {'cls': LLaMa2LLMBackbone, 'kwargs': {}}, 'mistral-v0.1-7b-pure': {'cls': MistralLLMBackbone, 'kwargs': {}}, 'mistral-v0.1-7b-instruct': {'cls': MistralLLMBackbone, 'kwargs': {}}, 'phi-2-3b': {'cls': PhiLLMBackbone, 'kwargs': {}}, 'qwen25-0_5b-pure': {'cls': Qwen25LLMBackbone, 'kwargs': {}}, 'qwen25-0_5b-extra': {'cls': Qwen25LLMBackbone, 'kwargs': {'num_extra_tokens': 256}}, 'qwen25-1_5b-pure': {'cls': Qwen25LLMBackbone, 'kwargs': {}}, 'qwen25-3b-pure': {'cls': Qwen25LLMBackbone, 'kwargs': {}}}

def get_vision_backbone_and_transform(vision_backbone_id: str, image_resize_strategy: str, image_sequence_len: int) -> Tuple[VisionBackbone, ImageTransform]:
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg['cls'](vision_backbone_id, image_resize_strategy, image_sequence_len=image_sequence_len, **vision_cfg['kwargs'])
        image_transform = vision_backbone.get_image_transform()
        return (vision_backbone, image_transform)
    else:
        raise ValueError(f'Vision Backbone `{vision_backbone_id}` is not supported!')

def get_llm_backbone_and_tokenizer(llm_backbone_id: str, llm_max_length: int=2048, hf_token: Optional[str]=None, runtime_mode: bool=False) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg['cls'](llm_backbone_id, llm_max_length=llm_max_length, hf_token=hf_token, runtime_mode=runtime_mode, **llm_cfg['kwargs'])
        tokenizer = llm_backbone.get_tokenizer()
        return (llm_backbone, tokenizer)
    else:
        raise ValueError(f'LLM Backbone `{llm_backbone_id}` is not supported!')

def get_vlm(model_id: str, arch_specifier: str, vision_backbone: VisionBackbone, llm_backbone: LLMBackbone, enable_mixed_precision_training: bool=True) -> PrismaticVLM:
    return PrismaticVLM(model_id, vision_backbone, llm_backbone, enable_mixed_precision_training=enable_mixed_precision_training, arch_specifier=arch_specifier)
