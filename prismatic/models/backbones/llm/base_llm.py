import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Sequence, Type
import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch
warnings.filterwarnings('ignore', category=FutureWarning)
overwatch = initialize_overwatch(__name__)

class LLMBackbone(nn.Module, ABC):

    def __init__(self, llm_backbone_id: str) -> None:
        super().__init__()
        self.identifier = llm_backbone_id
        self.llm: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizerBase = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        ...

    @abstractmethod
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> CausalLMOutputWithPast:
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> Type[nn.Module]:
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        ...

    @property
    @abstractmethod
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        ...

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

class HFCausalLLMBackbone(LLMBackbone, ABC):

    def __init__(self, llm_backbone_id: str, llm_family: str, llm_cls: Type[PreTrainedModel], hf_hub_path: str, llm_max_length: int=2048, hf_token: Optional[str]=None, runtime_mode: bool=False, use_flash_attention_2: bool=False) -> None:
        super().__init__(llm_backbone_id)
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.runtime_mode = runtime_mode
        if not self.runtime_mode:
            overwatch.info(f'Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]', ctx_level=1)
            self.llm = llm_cls.from_pretrained(hf_hub_path, token=hf_token, use_flash_attention_2=use_flash_attention_2 if not self.runtime_mode else False, do_sample=False, temperature=1.0, top_p=1.0)
        else:
            overwatch.info(f'Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]', ctx_level=1)
            llm_config = AutoConfig.from_pretrained(hf_hub_path, token=hf_token)
            if hasattr(llm_cls, 'from_config'):
                self.llm = llm_cls.from_config(llm_config)
            else:
                self.llm = llm_cls._from_config(llm_config)
        self.llm.config.use_cache = False if not self.runtime_mode else True
        if not self.runtime_mode:
            self.llm.enable_input_require_grads()
        overwatch.info(f'Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API', ctx_level=1)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_hub_path, model_max_length=self.llm_max_length, token=hf_token, padding_side='right')
        SPECIAL_CASES = {'phi-2-3b', 'qwen25-0_5b-pure', 'qwen25-0_5b-extra', 'qwen25-1_5b-pure', 'qwen25-3b-pure', 'qwen25-7b-pure'}
        if self.identifier in SPECIAL_CASES:
            return
        assert self.tokenizer('Test 123', add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id and self.tokenizer('Test 123', add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id, f'Default Tokenizer of type `{type(self.tokenizer)}` does not automatically prefix inputs with BOS token!\nPlease read the comment in `base_llm.py` for more information!'

    def get_fsdp_wrapping_policy(self) -> Callable:
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls})
        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> CausalLMOutputWithPast:
        output: CausalLMOutputWithPast = self.llm(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return output
