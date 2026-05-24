# Copyright 2023-2026 SGLang Team / SOAR-2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================================================
"""EAGLE3 draft head for MiniCPM-SALA.

Adapted from ``python/sglang/srt/models/llama_eagle3.py``. Two SALA-specific
adaptations relative to that template:

1. The decoder building blocks are SALA's dense components
   (``MiniCPMAttention``, ``MiniCPMMLP``, ``RMSNorm``) imported from
   ``sglang.srt.models.minicpm``. We deliberately do NOT reuse
   ``MiniCPMLightningMixer`` or any sparse-attention machinery here:
   the draft head is dense-only by design so that drafting is cheap
   and the (algorithmically complex) lightning-attention tree verify
   path is exercised only on the target model. Correctness of that
   target-side path is handled by the Phase 1 SimpleGLA tree-verify
   kernel (``fused_recurrent_simple_gla_update``).

2. ``MiniCPMHybridConfig`` carries SALA-specific keys (``mixer_types``,
   ``sparse_config``, ``lightning_*``) that are irrelevant to the draft
   head. We read only the standard transformer fields (``hidden_size``,
   ``num_attention_heads``, ``num_key_value_heads``, ``intermediate_size``,
   ``vocab_size``, ``rms_norm_eps``, ``rope_theta``, ``rope_scaling``,
   ``max_position_embeddings``, ``tie_word_embeddings``) plus the
   EAGLE3-specific ones (``draft_vocab_size``, ``target_hidden_size``).

Reference EAGLE3 paper:
  https://arxiv.org/abs/2503.01840  (Li et al., 2025)
Upstream cnets.py:
  https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""

from __future__ import annotations

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.minicpm import (
    MiniCPMAttention,
    MiniCPMMLP,
    MiniCPMSALAForCausalLM,
)
from sglang.srt.utils import add_prefix


class MiniCPMSALAEagle3DecoderLayer(nn.Module):
    """Single dense EAGLE3 draft decoder layer for MiniCPM-SALA.

    Structurally identical to ``llama_eagle3.LlamaDecoderLayer`` (the
    "midlayer"), with SALA-specific dense attention and MLP modules
    substituted. Inputs to ``qkv_proj`` are ``concat(embeds, hidden)``
    (i.e. ``2 * hidden_size``); ``hidden_norm`` is the EAGLE3-specific
    pre-norm on the target hidden states feed.
    """

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # EAGLE3-specific: separate pre-norm on the target hidden-state feed
        # before concatenation. See cnets.py decoder block.
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Build the dense attention block. We construct MiniCPMAttention
        # first then override its qkv_proj to accept 2*hidden_size input
        # (matching llama_eagle3's trick of feeding concat(embeds, hidden)).
        self.self_attn = MiniCPMAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            quant_config=quant_config,
            attn_use_rope=True,
            use_output_gate=getattr(config, "use_output_gate", False),
            prefix=add_prefix("self_attn", prefix),
        )
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("self_attn.qkv_proj", prefix),
        )

        self.mlp = MiniCPMMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mirror llama_eagle3.LlamaDecoderLayer.forward exactly.
        # The incoming ``residual`` argument is intentionally overwritten —
        # EAGLE3 always uses the pre-layer hidden_states as the residual
        # anchor for the single midlayer.
        residual = hidden_states

        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        # 2H concat — qkv_proj absorbs both signals jointly.
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Two-arg RMSNorm: returns (norm(x + residual), x + residual).
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniCPMSALAEagle3Model(nn.Module):
    """One-layer EAGLE3 backbone for SALA.

    Mirrors ``llama_eagle3.LlamaModel``. The forward consumes the target
    model's auxiliary hidden states (typically the concatenation of three
    intermediate layers, hence ``hidden_size_in * 3`` for the input
    projection) and produces (hidden_states_to_logits, [hidden_states_to_aux]).
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # EAGLE3 fuses three target hidden-state layers as input — see
        # the upstream paper §3.2 ("multi-layer fusion"). The "*3" matches
        # llama_eagle3.LlamaModel.__init__ and is the load-bearing channel
        # count expected by the trained draft-head weights.
        self.hidden_size_in = getattr(
            config, "target_hidden_size", config.hidden_size
        )
        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.midlayer = MiniCPMSALAEagle3DecoderLayer(
            config, layer_id=0, quant_config=quant_config, prefix=prefix
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Tuple[torch.Tensor, list]:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # Idle batch path: an empty hidden_states means the scheduler had
        # nothing to draft this step; return cleanly so callers can skip
        # the LogitsProcessor.
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        # Final fused norm + residual add. The *pre*-norm sum (hidden +
        # residual) is what we surface as the auxiliary hidden state, so
        # downstream tree-prediction / training losses can supervise the
        # midlayer's contribution directly.
        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        return hidden_states_to_logits, [hidden_states_to_aux]


class MiniCPMSALAEagle3ForCausalLM(MiniCPMSALAForCausalLM):
    """EAGLE3 entry class for MiniCPM-SALA.

    Inherits from ``MiniCPMSALAForCausalLM`` so the loader's
    ``isinstance`` checks for ``MiniCPMSALAForCausalLM`` (used by the
    SALA-aware code paths in the scheduler / model_runner) continue to
    succeed for a draft model. We override ``__init__``, ``forward``,
    and ``load_weights`` — the parent's heavy machinery (sparse config
    plumbing etc.) is not exercised because the draft is dense-only.

    Hot-token-id remapping ``d2t`` is supported (the draft can be
    trained against a reduced ``draft_vocab_size``); ``t2d`` is silently
    ignored at load time per upstream convention.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Skip parent's __init__ — we don't want MiniCPMModel built.
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.config.num_hidden_layers != 1:
            raise ValueError(
                "MiniCPM-SALA EAGLE3 currently only supports num_hidden_layers=1; "
                f"got {self.config.num_hidden_layers}. Multi-layer drafting can be "
                "added later but is out of scope for this port."
            )

        self.model = MiniCPMSALAEagle3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # LM head wiring (identical to llama_eagle3):
        # - If tie_word_embeddings: lm_head is the draft's own embed_tokens.
        # - Else if draft_vocab_size is None: load lm_head from target weights
        #   (full vocab_size).
        # - Else: separate ParallelLMHead sized for the reduced draft vocab.
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = getattr(config, "draft_vocab_size", None)
            if draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
                draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        # LogitsProcessor uses the draft's (possibly reduced) vocab size.
        config_ = copy.deepcopy(config)
        config_.vocab_size = config_.draft_vocab_size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id: Optional[torch.Tensor] = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Note: unlike MiniCPMSALAForCausalLM.forward we do NOT apply
        # scale_emb on input embeds or scale_width on hidden_states.
        # The draft head's parameters absorb scale during training.
        hidden_states_to_logits, _aux_list = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states_to_logits, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> None:
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            # EAGLE3 hot-token mapping: ``d2t[i]`` is the offset of draft id
            # i from its corresponding target id. Adding ``arange`` produces
            # the absolute target id, cached on the model for the sampler
            # to remap accepted draft tokens at runtime.
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(
                    loaded_weight.shape[0]
                )
                continue

            # ``t2d`` is the inverse mapping; the draft model itself does
            # not consume it.
            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                resolved = (
                    f"model.{name}" if name not in params_dict else name
                )
                if resolved in params_dict:
                    param = params_dict[resolved]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved = name if name in params_dict else f"model.{name}"
                if resolved in params_dict:
                    param = params_dict[resolved]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_hot_token_id(self) -> Optional[torch.Tensor]:
        return self.hot_token_id


EntryClass = [MiniCPMSALAEagle3ForCausalLM]
