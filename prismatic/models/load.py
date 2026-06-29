import json
import os
from pathlib import Path
from typing import List, Optional, Union
from huggingface_hub import HfFileSystem, hf_hub_download
from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS, ActionTokenizer
overwatch = initialize_overwatch(__name__)
HF_HUB_REPO = 'TRI-ML/prismatic-vlms'
VLA_HF_HUB_REPO = 'openvla/openvla-dev'

def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())

def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())

def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `model_id_or_name = {model_id_or_name!r}; check `prismatic.available_model_names()`")
    print(json.dumps((description := GLOBAL_REGISTRY[model_id_or_name]['description']), indent=2))
    return description

def load(model_id_or_path: Union[str, Path], hf_token: Optional[str]=None, cache_dir: Optional[Union[str, Path]]=None, load_for_training: bool=False, image_sequence_len: Optional[int]=None) -> PrismaticVLM:
    if os.path.isdir(model_id_or_path):
        overwatch.info(f'Loading from local path `{(run_dir := Path(model_id_or_path))}`')
        (config_json, checkpoint_pt) = (run_dir / 'config.json', run_dir / 'checkpoints' / 'step-020792-epoch-01-loss=0.5268.pt')
        assert config_json.exists(), f'Missing `config.json` for `run_dir = {run_dir!r}`'
        assert checkpoint_pt.exists(), f'Missing checkpoint for `run_dir = {run_dir!r}`'
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `model_id_or_path = {model_id_or_path!r}; check `prismatic.available_model_names()`")
        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f'{model_id}/config.json', cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(repo_id=HF_HUB_REPO, filename=f'{model_id}/checkpoints/latest-checkpoint.pt', cache_dir=cache_dir)
    with open(config_json, 'r') as f:
        model_cfg = json.load(f)['model']
    overwatch.info(f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]")
    if image_sequence_len is None:
        if hasattr(model_cfg, 'image_sequence_len'):
            image_sequence_len = model_cfg.image_sequence_len
        else:
            image_sequence_len = 1
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    (vision_backbone, image_transform) = get_vision_backbone_and_transform(model_cfg['vision_backbone_id'], model_cfg['image_resize_strategy'], image_sequence_len)
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    (llm_backbone, tokenizer) = get_llm_backbone_and_tokenizer(model_cfg['llm_backbone_id'], llm_max_length=model_cfg.get('llm_max_length', 2048), hf_token=hf_token, runtime_mode=not load_for_training)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(checkpoint_pt, model_cfg['model_id'], vision_backbone, llm_backbone, arch_specifier=model_cfg['arch_specifier'], freeze_weights=not load_for_training)
    return vlm

def load_vla(model_id_or_path: Union[str, Path], hf_token: Optional[str]=None, cache_dir: Optional[Union[str, Path]]=None, load_for_training: bool=False, step_to_load: Optional[int]=None, model_type: str='pretrained', image_sequence_len: Optional[int]=None) -> OpenVLA:
    if os.path.isfile(model_id_or_path):
        overwatch.info(f'Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`')
        assert checkpoint_pt.suffix == '.pt' and checkpoint_pt.parent.name == 'checkpoints', 'Invalid checkpoint!'
        run_dir = checkpoint_pt.parents[1]
        (config_json, dataset_statistics_json) = (run_dir / 'config.json', run_dir / 'dataset_statistics.json')
        assert config_json.exists(), f'Missing `config.json` for `run_dir = {run_dir!r}`'
        assert dataset_statistics_json.exists(), f'Missing `dataset_statistics.json` for `run_dir = {run_dir!r}`'
    else:
        overwatch.info(f'Checking HF for `{(hf_path := str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path))}`')
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `hf_path = {hf_path!r}`")
        step_to_load = f'{step_to_load:06d}' if step_to_load is not None else None
        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/step-{(step_to_load if step_to_load is not None else '')}*.pt")
        if len(valid_ckpts) == 0 or (step_to_load is not None and len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")
        target_ckpt = Path(valid_ckpts[-1]).name
        overwatch.info(f'Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`')
        with overwatch.local_zero_first():
            relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(repo_id=VLA_HF_HUB_REPO, filename=f"{relpath / 'config.json'!s}", cache_dir=cache_dir)
            dataset_statistics_json = hf_hub_download(repo_id=VLA_HF_HUB_REPO, filename=f"{relpath / 'dataset_statistics.json'!s}", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(repo_id=VLA_HF_HUB_REPO, filename=f"{relpath / 'checkpoints' / target_ckpt!s}", cache_dir=cache_dir)
    with open(config_json, 'r') as f:
        vla_cfg = json.load(f)['vla']
        base_vlm = vla_cfg['base_vlm']
    if os.path.isdir(base_vlm):
        with open(Path(base_vlm) / 'config.json', 'r') as f:
            base_cfg = json.load(f)['model']
            base_vlm = base_cfg['model_id']
    overwatch.info(f'Base vlm: {base_vlm}')
    model_cfg = ModelConfig.get_choice_class(base_vlm)()
    with open(dataset_statistics_json, 'r') as f:
        norm_stats = json.load(f)
    overwatch.info(f'Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]')
    if image_sequence_len is None:
        if hasattr(model_cfg, 'image_sequence_len'):
            image_sequence_len = model_cfg.image_sequence_len
        else:
            image_sequence_len = 1
    overwatch.info(f'Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]')
    (vision_backbone, image_transform) = get_vision_backbone_and_transform(model_cfg.vision_backbone_id, model_cfg.image_resize_strategy, image_sequence_len)
    overwatch.info(f'Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers')
    (llm_backbone, tokenizer) = get_llm_backbone_and_tokenizer(model_cfg.llm_backbone_id, llm_max_length=model_cfg.llm_max_length, hf_token=hf_token, runtime_mode=not load_for_training)
    ac_tokenizer = vla_cfg['action_tokenizer'] if 'action_tokenizer' in vla_cfg else 'action_tokenizer'
    action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS[ac_tokenizer](llm_backbone.get_tokenizer())
    overwatch.info(f'Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint')
    vla = OpenVLA.from_pretrained(checkpoint_pt, model_cfg.model_id, vision_backbone, llm_backbone, arch_specifier=model_cfg.arch_specifier, freeze_weights=not load_for_training, norm_stats=norm_stats, action_tokenizer=action_tokenizer)
    return vla
