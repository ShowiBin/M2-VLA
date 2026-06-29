import json
from functools import partial
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from prismatic.overwatch.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

class ActionTokenizer:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, bins: int=256, min_action: int=-1, max_action: int=1, use_extra: bool=False) -> None:
        (self.tokenizer, self.n_bins, self.min_action, self.max_action) = (tokenizer, bins, min_action, max_action)
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError('Cannot use extra tokens for this tokenizer!')
        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action: np.ndarray, use_minivlm) -> Union[str, List[str]]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        if use_minivlm:
            return (self.tokenizer_len - discretized_action).tolist()
        elif len(discretized_action.shape) <= 1:
            return self.tokenizer.decode(list(self.tokenizer_len - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer_len - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer_len - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    @property
    def required_future_horizon(self) -> int:
        return 0

class VQActionTokenizer(ActionTokenizer):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, vq_vae_path='', device='cpu', use_extra: bool=False):
        self.tokenizer = tokenizer
        self.device = device
        from vqvae.vqvae import VqVae
        self.vq_path = Path(vq_vae_path)
        assert self.vq_path.exists(), f'Missing VQ VAE path: {self.vq_path}'
        vq_model_path = self.vq_path / 'checkpoints' / 'model.pt'
        vq_config_path = self.vq_path / 'config.json'
        assert vq_model_path.exists(), f'Missing VQ checkpoint path: {vq_model_path}'
        assert vq_config_path.exists(), f'Missing VQ config path: {vq_config_path}'
        with open(vq_config_path, 'r') as f:
            vq_config = dict(json.load(f))
        vq_config['load_dir'] = vq_model_path
        vq_config['eval'] = True
        vq_config['device'] = self.device
        overwatch.info(f'Loading VQ VAE for Action Tokenization from {vq_config_path}...')
        self.vq_vae = VqVae(**vq_config)
        overwatch.info(f'Found VQ VAE parameters: \n{self.vq_vae}')
        self.n_bins = self.vq_vae.vqvae_n_embed
        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError('Cannot use extra tokens for this tokenizer!')
        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        action = torch.from_numpy(action).to(self.device).reshape((1, self.vq_vae.input_dim_h, self.vq_vae.input_dim_w))
        (_, vq_code) = self.vq_vae.get_code(action)
        assert torch.all(vq_code >= 0) and torch.all(vq_code < self.n_bins)
        return self.tokenizer.decode(list(self.tokenizer_len - 1 - vq_code[0].numpy()))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        action_token_ids = self.tokenizer_len - 1 - action_token_ids
        initial_shape = action_token_ids.shape
        action_token_ids = np.clip(action_token_ids, 0, self.n_bins - 1)
        action_token_ids = torch.from_numpy(action_token_ids).to(self.device).reshape(-1, self.vq_vae.vqvae_groups)
        assert torch.all(action_token_ids >= 0) and torch.all(action_token_ids < self.n_bins)
        latent = self.vq_vae.draw_code_forward(action_token_ids)
        ret_action = self.vq_vae.get_action_from_latent(latent)
        if action_token_ids.shape[0] == 1 and len(initial_shape) == 1:
            return ret_action[0, 0]
        return ret_action[:, 0]

    @property
    def required_future_horizon(self) -> int:
        return self.vq_vae.input_dim_h - 1
ACTION_TOKENIZERS = {'action_tokenizer': ActionTokenizer, 'extra_action_tokenizer': partial(ActionTokenizer, use_extra=True), 'libero_vq_action_tokenizer': partial(VQActionTokenizer, vq_vae_path='vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512'), 'libero_vq_extra_action_tokenizer': partial(VQActionTokenizer, vq_vae_path='vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512', use_extra=True), 'libero_vq_h0_extra_action_tokenizer': partial(VQActionTokenizer, vq_vae_path='vq/pretrain_vq+mx-libero_90+fach-0+ng-7+nemb-128+nlatent-512', use_extra=True), 'bridge_vq_extra_action_tokenizer': partial(VQActionTokenizer, vq_vae_path='vq/pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512', use_extra=True)}
