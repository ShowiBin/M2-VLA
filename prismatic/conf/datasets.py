from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple
from draccus import ChoiceRegistry

@dataclass
class DatasetConfig(ChoiceRegistry):
    dataset_id: str
    align_stage_components: Tuple[Path, Path]
    finetune_stage_components: Tuple[Path, Path]
    dataset_root_dir: Path

@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = 'llava-v15'
    align_stage_components: Tuple[Path, Path] = (Path('download/llava-laion-cc-sbu-558k/chat.json'), Path('download/llava-laion-cc-sbu-558k/'))
    finetune_stage_components: Tuple[Path, Path] = (Path('download/llava-v1.5-instruct/llava_v1_5_mix665k.json'), Path('download/llava-v1.5-instruct/'))
    dataset_root_dir: Path = Path('/mnt/fsx/skaramcheti/datasets/prismatic-vlms')

@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = 'llava-multimodal'
    align_stage_components: Tuple[Path, Path] = (Path('download/llava-laion-cc-sbu-558k/chat.json'), Path('download/llava-laion-cc-sbu-558k/'))
    finetune_stage_components: Tuple[Path, Path] = (Path('download/llava-v1.5-instruct/llava_v1_5_stripped625k.json'), Path('download/llava-v1.5-instruct/'))
    dataset_root_dir: Path = Path('/mnt/fsx/skaramcheti/datasets/prismatic-vlms')

@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = 'llava-lvis4v'
    align_stage_components: Tuple[Path, Path] = (Path('download/llava-laion-cc-sbu-558k/chat.json'), Path('download/llava-laion-cc-sbu-558k/'))
    finetune_stage_components: Tuple[Path, Path] = (Path('download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json'), Path('download/llava-v1.5-instruct/'))
    dataset_root_dir: Path = Path('/mnt/fsx/skaramcheti/datasets/prismatic-vlms')

@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = 'llava-lrv'
    align_stage_components: Tuple[Path, Path] = (Path('download/llava-laion-cc-sbu-558k/chat.json'), Path('download/llava-laion-cc-sbu-558k/'))
    finetune_stage_components: Tuple[Path, Path] = (Path('download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json'), Path('download/llava-v1.5-instruct/'))
    dataset_root_dir: Path = Path('/mnt/fsx/skaramcheti/datasets/prismatic-vlms')

@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = 'llava-lvis4v-lrv'
    align_stage_components: Tuple[Path, Path] = (Path('download/llava-laion-cc-sbu-558k/chat.json'), Path('download/llava-laion-cc-sbu-558k/'))
    finetune_stage_components: Tuple[Path, Path] = (Path('download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json'), Path('download/llava-v1.5-instruct/'))
    dataset_root_dir: Path = Path('/mnt/fsx/skaramcheti/datasets/prismatic-vlms')

@unique
class DatasetRegistry(Enum):
    LLAVA_V15 = LLaVa_V15_Config
    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config
    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config
    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
