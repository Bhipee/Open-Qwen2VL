"""
qwen3_moe.py

Class definition for all LLMs derived from Qwen3MoeForCausalLM. Note that transformer layer of this model will be directly wrapped by fsdp, thus less efficient than megablock-based version.
"""
from typing import (
    Callable,
    Type,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import torch
from torch import nn as nn
from transformers import Qwen3MoeForCausalLM
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    Qwen2PromptBuilder,
)

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
QWEN3MoE_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "qwen3-30b-a3b": {
        "llm_family": "qwen3_moe", "llm_cls": Qwen3MoeForCausalLM, "hf_hub_path": "Qwen3-30B-A3B"
    },
}
# fmt: on


class Qwen3MoELLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 4096,
        mount_path: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            mount_path=mount_path,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **QWEN3MoE_MODELS[llm_backbone_id],
        )

        # [Special Case] Qwen-2.5 PAD Token Handling --> for clarity, we add an extra token, no need to resize the model embedding layer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<s>"]})
        self.tokenizer.bos_token = "<s>"

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Qwen2PromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen3MoeDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16