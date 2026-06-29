from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union
from draccus import ChoiceRegistry

@dataclass
class VLAConfig(ChoiceRegistry):
    vla_id: str
    base_vlm: Union[str, Path]
    freeze_vision_backbone: bool
    freeze_llm_backbone: bool
    unfreeze_last_llm_layer: bool
    data_mix: str
    shuffle_buffer_size: int
    epochs: int
    max_steps: Optional[int]
    save_every_n_steps: Optional[int]
    expected_world_size: int
    global_batch_size: int
    per_device_batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    warmup_ratio: float
    train_strategy: str
    action_tokenizer: str
    image_sequence_len: int
    use_wrist_image: bool
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True

@dataclass
class Exp_SigLIP_224px_Bridge(VLAConfig):
    vla_id: str = 'siglip-224px+mx-bridge'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    image_sequence_len: int = 1
    use_wrist_image: bool = False
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False
    data_mix: str = 'bridge'
    shuffle_buffer_size: int = 256000
    epochs: int = 1000
    max_steps: Optional[int] = None
    save_every_n_steps: Optional[int] = 25000
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32
    learning_rate: float = 2e-05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = 'constant'
    warmup_ratio: float = 0.0
    train_strategy: str = 'fsdp-full-shard'
    action_tokenizer: str = 'action_tokenizer'

@dataclass
class Exp_FreezeVIT_SigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px-icy+mx-bridge'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    freeze_vision_backbone: bool = True

@dataclass
class Exp_DinoSigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'prism-dinosiglip-224px+mx-bridge'
    base_vlm: Union[str, Path] = 'prism-dinosiglip-224px+7b'
    data_mix: str = 'bridge'

@dataclass
class Exp_SigLIP_224px_OXE_Magic_Soup(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px+mx-oxe-magic-soup'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    data_mix: str = 'oxe_magic_soup'
    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32

@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'prism-qwen25-dinosiglip-224px+0_5b+mx-oxe-magic-soup'
    base_vlm: Union[str, Path] = 'prism-qwen25-extra-dinosiglip-224px+0_5b'
    data_mix: str = 'oxe_magic_soup'
    action_tokenizer: str = 'extra_action_tokenizer'
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup):
    vla_id: str = 'prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90'
    data_mix: str = 'libero_90'
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

@dataclass
class Exp_Qwen25_DinoSigLIP_224px_T2_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90):
    vla_id: str = 'prism-qwen25-dinosiglip-224px-t2+0_5b+mx-libero-90'
    image_sequence_len: int = 2

@dataclass
class Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90):
    vla_id: str = 'prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90'
    image_sequence_len: int = 2
    use_wrist_image: bool = True

@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'prism-qwen25-dinosiglip-224px+0_5b+mx-bridge'
    base_vlm: Union[str, Path] = 'prism-qwen25-extra-dinosiglip-224px+0_5b'
    data_mix: str = 'bridge_dataset'
    action_tokenizer: str = 'extra_action_tokenizer'
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

@dataclass
class Exp_DinoSigLIP_224px_LIBERO_90(Exp_DinoSigLIP_224px_Bridge):
    vla_id: str = 'prism-dinosiglip-224px+mx-libero-90'
    data_mix: str = 'libero_90'
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

@dataclass
class Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'prism-dinosiglip-224px+mx-oxe-magic-soup-plus'
    base_vlm: Union[str, Path] = 'prism-dinosiglip-224px+7b'
    data_mix: str = 'oxe_magic_soup_plus_minus'
    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32

@dataclass
class Exp_SigLIP_224px_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px+mx-tdroid_carrot_in_bowl'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    data_mix: str = 'tdroid_carrot_in_bowl'

@dataclass
class Exp_SigLIP_224px_TDROID_PourCornInPot(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px+mx-tdroid_pour_corn_in_pot'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    data_mix: str = 'tdroid_pour_corn_in_pot'

@dataclass
class Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px-icy+mx-tdroid_carrot_in_bowl'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    data_mix: str = 'tdroid_carrot_in_bowl'

@dataclass
class Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px-last_layer+mx-tdroid_carrot_in_bowl'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True
    data_mix: str = 'tdroid_carrot_in_bowl'

@dataclass
class Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px-sandwich+mx-tdroid_carrot_in_bowl'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True
    data_mix: str = 'tdroid_carrot_in_bowl'

@dataclass
class Exp_SigLIP_224px_Droid_Wipe(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'siglip-224px+mx-droid_wipe'
    base_vlm: Union[str, Path] = 'siglip-224px+7b'
    data_mix: str = 'droid_wipe'

@unique
class VLARegistry(Enum):
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge
    DINOSIGLIP_224PX_MX_BRIDGE = Exp_DinoSigLIP_224px_Bridge
    DINOSIGLIP_224PX_MX_LIBERO_90 = Exp_DinoSigLIP_224px_LIBERO_90
    FREEZE_SIGLIP_224PX_MX_BRIDGE = Exp_FreezeVIT_SigLIP_224px_Bridge
    SIGLIP_224PX_MX_OXE_MAGIC_SOUP = Exp_SigLIP_224px_OXE_Magic_Soup
    DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS = Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus
    QWEN25_DINOSIGLIP_224PX_0_5B_MX_OXE_MAGIC_SOUP = Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup
    QWEN25_DINOSIGLIP_224PX_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90
    QWEN25_DINOSIGLIP_224PX_T2_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_T2_0_5B_LIBERO_90
    QWEN25_DINOSIGLIP_224PX_WRIST_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_LIBERO_90
    QWEN25_DINOSIGLIP_224PX_0_5B_BRIDGE = Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge
    SIGLIP_224PX_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_TDROID_CarrotInBowl
    SIGLIP_224PX_MX_TDROID_POUR_CORN_IN_POT = Exp_SigLIP_224px_TDROID_PourCornInPot
    SIGLIP_224PX_ICY_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl
    SIGLIP_224PX_LASTLAYER_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl
    SIGLIP_224PX_SANDWICH_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl
    SIGLIP_224PX_MX_DROID_WIPE = Exp_SigLIP_224px_Droid_Wipe

    @property
    def vla_id(self) -> str:
        return self.value.vla_id
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
