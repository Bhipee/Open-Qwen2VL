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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, _module_wrap_policy
from functools import partial
import transformers
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention, Qwen3MoeRMSNorm
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    Qwen2PromptBuilder,
)

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
QWEN3MoE_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "qwen3-30b-a3b-megablock": {
        "llm_family": "qwen3_moe", "llm_cls": Qwen3MoeForCausalLM, "hf_hub_path": "Qwen3-30B-A3B"
    },
}
# fmt: on


class Qwen3MoEMegaBlockLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 4096,
        mount_path: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        # monkey patch
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeDecoderLayer = Qwen3MoeDecoderLayer
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
        return Qwen3MoeDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    # def get_fsdp_wrapping_policy(self) -> Callable:
    #     """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
    #    wrap_policy = partial(_module_wrap_policy, module_classes={Qwen3MoeDecoderLayer})
    #     return wrap_policy

#### MOE ####
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            # self.mlp = Qwen3MoeSparseMoeBlock(config)
            try:
                from megablocks.layers.moe import MoE
                from megablocks.layers.arguments import Arguments as MoEArgs
            except ImportError:
                import logging
                logging.warning(f"Megablocks not installed. To train MoE, install with pip install megablocks.")
            moe_args = MoEArgs(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.intermediate_size if config.intermediate_size is not None else config.hidden_size * 4,
                moe_num_experts=config.num_experts,
                # not sure
                mlp_impl="grouped",
                moe_expert_model_parallelism=False,##### currently fixed
                moe_top_k=config.num_experts_per_tok,
                moe_capacity_factor=1.25,##### currently fixed
                moe_loss_weight=0.1,##### currently fixed
                # device=torch.cuda.current_device(),
                device='meta',
                bf16=False,
                fp16=False,
            )
            self.mlp = MoE(moe_args)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

# class Qwen3MoeDecoderLayer(Qwen2MoeDecoderLayer, nn.Module):
#     def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
#         nn.Module().__init__()
#         self.hidden_size = config.hidden_size

#         self.self_attn = Qwen3MoeAttention(config, layer_idx)

#         self.layer_idx = layer_idx
#         self.mlp_only_layers = config.mlp_only_layers
#         if (layer_idx not in config.mlp_only_layers) and (
#             config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
#         ):
#             try:
#                 from megablocks.layers.moe import MoE
#                 from megablocks.layers.arguments import Arguments as MoEArgs
#             except ImportError:
#                 import logging
#                 logging.warning(f"Megablocks not installed. To train MoE, install with pip install megablocks.")
#             # self.mlp = Qwen3MoeSparseMoeBlock(config)
#             moe_args = MoEArgs(
#                 hidden_size=config.hidden_size,
#                 ffn_hidden_size=config.intermediate_size if config.intermediate_size is not None else config.hidden_size * 4,
#                 moe_num_experts=config.num_experts,
#                 # handled by fsdp
#                 moe_weight_parallelism=False,##### currently fixed
#                 moe_expert_model_parallelism=False,##### currently fixed
#                 moe_top_k=config.num_experts_per_tok,
#                 moe_capacity_factor=1.25,##### currently fixed
#                 moe_loss_weight=0.1,##### currently fixed
#                 device=torch.cuda.current_device(),
#                 bf16=False,
#                 fp16=False,
#             )
#             self.mlp = MoE(moe_args)
#         else:
#             self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

#         self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: Optional[bool] = False,
#         output_router_logits: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
#                 `(batch, sequence_length)` where padding elements are indicated by 0.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             output_router_logits (`bool`, *optional*):
#                 Whether or not to return the logits of all the routers. They are useful for computing the router loss,
#                 and should not be returned during inference.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#             cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#                 Indices depicting the position of the input sequence tokens in the sequence.
#             position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
#                 Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
#                 with `head_dim` being the embedding dimension of each attention head.
#             kwargs (`dict`, *optional*):
#                 Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
#                 into the model
#         """

#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             position_embeddings=position_embeddings,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)

#         if self.layer_idx in self.mlp_only_layers:
#             hidden_states, _ = self.mlp(hidden_states)
#         else:
#             hidden_states = self.mlp(hidden_states)
#         if isinstance(hidden_states, tuple):
#             hidden_states, router_logits = hidden_states
#         else:
#             router_logits = None

#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if output_router_logits:
#             outputs += (router_logits,)

#         return outputs

