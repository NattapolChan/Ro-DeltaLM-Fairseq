# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoderBase,
    TransformerEncoderBase,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)

from fairseq.modules.rotary_positional_embedding import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
)
import math

from fairseq.modules.espnet_multihead_attention import (
    RotaryPositionalEmbedding,
    RotaryPositionMultiHeadedAttention
)


from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase
)
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq import utils
from fairseq.file_io import PathManager
import logging
logger = logging.getLogger(__name__)


def upgrade_state_dict_for_deltalm(
    state_dict: Dict[str, Any], pretrained_deltalm_checkpoint: str, is_encoder=True,
) -> Dict[str, Any]:
    """
    update state dict when checkpoint condition is met
    """
    if not os.path.exists(pretrained_deltalm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_deltalm_checkpoint))

    with open(pretrained_deltalm_checkpoint, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    deltalm_state_dict = state["weights"]

    new_deltalm_state_dict = {}

    for key in deltalm_state_dict.keys():
        if is_encoder:
            if key.startswith('encoder.') or key.startswith('src_embedding.'):
                new_key = key.replace('encoder.', '')
                new_key = new_key.replace('src_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]
        else:
            if key.startswith('decoder.') or key.startswith('tgt_embedding.'):
                new_key = key.replace('decoder.', '')
                new_key = new_key.replace('tgt_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]
    

    deltalm_state_dict = new_deltalm_state_dict
    
    for key in deltalm_state_dict.keys():
        map_key = key
        map_key = map_key.replace('.ffn_1.fc1', '.fc3')
        map_key = map_key.replace('.ffn_1.fc2', '.fc4')
        map_key = map_key.replace('.ffn_2', '')
        map_key = map_key.replace('.ffn.', '.')

        # if not is_encoder:
        #     map_key = map_key.replace('.self_attn.q_proj.', '.self_rotary.linear_q.')
        #     map_key = map_key.replace('.self_attn.k_proj.', '.self_rotary.linear_k.')
        #     map_key = map_key.replace('.self_attn.v_proj.', '.self_rotary.linear_v.')
        #     map_key = map_key.replace('.self_attn.out_proj.', '.self_rotary.linear_out.')
        
        print(map_key)
        map_key = map_key.replace('emb_layer_norm', 'layernorm_embedding')
        # assert map_key in state_dict, map_key
        if map_key not in state_dict:
            print("[WARNING] " + map_key+ "not in state_dict")
            continue
        if 'embed_positions' in key or 'embed_tokens' in key:
            left_size = state_dict[map_key].size(0)
            right_size = deltalm_state_dict[key].size(0)
            if left_size <= right_size:
                state_dict[map_key] = deltalm_state_dict[key][:left_size]
            else:
                state_dict[map_key][:right_size] = deltalm_state_dict[key]
        else:
            state_dict[map_key] = deltalm_state_dict[key]

    return state_dict


@register_model("testdeltalm")
class DeltaLMModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-deltalm-checkpoint",
            type=str,
            metavar="STR",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        return DeltaLMEncoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return DeltaLMDecoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)


class DeltaLMEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "pretrained_deltalm_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self.state_dict(),
                pretrained_deltalm_checkpoint=args.pretrained_deltalm_checkpoint,
                is_encoder=True,
            )
            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            print(f"encoder/embed_tokens: "+ str(float(torch.flatten(self.embed_tokens.weight)[0])))
            print(f"encoder/embed_positions: "+ str(float(torch.flatten(self.embed_positions.weight)[0])))

            for idx, each_module in enumerate(self.layers.children()):
                print(f"encoder/self_attn/{idx}/q_proj: " + str(float(torch.flatten(each_module.self_attn.q_proj.weight)[0])))
                print(f"encoder/self_attn/{idx}/k_proj: " + str(float(torch.flatten(each_module.self_attn.k_proj.weight)[0])))
                print(f"encoder/self_attn/{idx}/v_proj: " + str(float(torch.flatten(each_module.self_attn.v_proj.weight)[0])))
                print(f"decoder/self_attn/{idx}/out_proj: " + str(float(torch.flatten(each_module.self_attn.out_proj.weight)[0])))

                print(f"encoder/self_attn_layer_norm: "+ str(float(torch.flatten(each_module.self_attn_layer_norm.weight)[0])))
                print(f"encoder/fc1: "+ str(float(torch.flatten(each_module.fc1.weight)[0])))
                print(f"encoder/fc2: "+ str(float(torch.flatten(each_module.fc2.weight)[0])))

            logger.info("Load DeltaLM's encoder from {0}".format(args.pretrained_deltalm_checkpoint))

class DeltaLMDecoder(TransformerDecoderBase):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, "pretrained_deltalm_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self.state_dict(),
                pretrained_deltalm_checkpoint=args.pretrained_deltalm_checkpoint,
                is_encoder=False,
            )
            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            
            logger.info("Load DeltaLM's decoder from {0}".format(args.pretrained_deltalm_checkpoint))
            # for key in self.state_dict():
                # print(key + " " + str(float(torch.flatten(self.state_dict()[key])[0])))

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = DeltaLMDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

class DeltaLMDecoderLayer(TransformerDecoderLayerBase):
    
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayerBase, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        # self.self_rotary = RotaryPositionMultiHeadedAttention(
        #     self.embed_dim,
        #     args.decoder.attention_heads,
        #     args.dropout,
        #     precision=torch.float32,
        #     rotary_emd_base=1e-7
        # )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )

        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    
    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return RotaryMultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=0, #cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.decoder.xformers_att_config,
        )

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        src_lang_id = None,
        tgt_lang_id = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        return x, attn, None


    
class RotaryMultiheadAttention(MultiheadAttention):
    def __init__(
            self, embed_dim, num_heads, kdim=None, vdim=None, 
            dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, 
            self_attention=False, encoder_decoder_attention=False, q_noise=0, 
            qn_block_size=8, xformers_att_config: str | None = None, 
            xformers_blocksparse_layout: Tensor | None = None, 
            xformers_blocksparse_blocksize: int | None = 16
    ):
        super().__init__(embed_dim, num_heads, kdim, vdim, 
                         dropout, bias, add_bias_kv, add_zero_attn, 
                         self_attention, encoder_decoder_attention, 
                         q_noise, qn_block_size, xformers_att_config, 
                         xformers_blocksparse_layout, xformers_blocksparse_blocksize)
        # self.rotary_ndims = self.d_k  # also try self.d_k//2
        rotary_emd_base = 1e-9
        self.d_k = embed_dim // num_heads
        self.rotary_emb = RotaryPositionalEmbedding(
            self.d_k, base=rotary_emd_base, precision=torch.float32
        )
        self.dropout = nn.Dropout(p=dropout)
        self.h = num_heads
    
    def forward_rotary(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute rotary position attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        Notes:
            Assumes self attn
        """

        T, B, C = value.size()
        query = query.view(T, B, self.h, self.d_k)
        key = key.view(T, B, self.h, self.d_k)
        value = value.view(T, B, self.h, self.d_k)
        cos, sin = self.rotary_emb(value, seq_len=T)
        query, key = apply_rotary_pos_emb(
            query, key, cos, sin, offset=0
        )  # offset is based on layer_past

        query = query.view(T, B, self.h * self.d_k)
        key = key.view(T, B, self.h * self.d_k)
        value = value.view(T, B, self.h * self.d_k)

        # TBD to BTD
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None

    def forward_qkv(self, query, key, value, **kwargs):
        """Transform query, key and value.
        Args:
            query: Query tensor  B X T1 X C
            key: Key tensor B X T2 X C
            value: Value tensor  B X T2 X C
        Returns:
            torch.Tensor: Transformed query tensor  B X n_head X T1 X d_k
            torch.Tensor: Transformed key tensor B X n_head X T2 X d_k
            torch.Tensor: Transformed value tensor  B X n_head X T2 X d_k
        """
        n_batch = query.size(0)

        q = self.q_proj(query).view(n_batch, -1, self.h, self.d_k)
        k = self.k_proj(key).view(n_batch, -1, self.h, self.d_k)
        v = self.v_proj(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value: Transformed value B X n_head X T2 X d_k.
            scores: Attention score  B X n_head X T1 X T2
            mask: Mask  T2 X B
        Returns:
            torch.Tensor: Transformed value  B X T1 X d_model
                weighted by the attention score  B X T1 X T2
        """
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(bool),
                float("-inf"),  # (batch, head, time1, time2)
            )
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.out_proj(x)  # (batch, time1, d_model)
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            and not self.skip_embed_dim_check
        ):
            assert key is not None and value is not None

            if self.use_xformers:
                print("WARNING | !!")
                return self._xformers_attn_forward(
                    query, key, value, key_padding_mask, need_weights, attn_mask
                )

            else:
                out1 = self.forward_rotary(
                    query, 
                    key, 
                    value, 
                    key_padding_mask.bool() if key_padding_mask is not None else None
                )
                out2 = F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask.bool() if key_padding_mask is not None else None,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                )
                # print("out1__ : ")
                # print(torch.flatten(out1[0])[:3])
                # print("out2__ : ")
                # print(torch.flatten(out2[0])[:3])
                # print("proj_q : " + self.q_proj.bias)
                # print("proj_k : " + self.k_proj.bias)
                # print("proj_v : " + self.v_proj.bias)

                return out1

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                        :, :, 0, :
                    ]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1)
                        )[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            print("e")
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz
            )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    kv_bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                kv_bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]),
                k.view((kv_bsz, self.num_heads) + k.size()[1:]),
            )
            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.view(
                    kv_bsz, -1, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn: Optional[Tensor] = None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.view(
                    (
                        kv_bsz,
                        -1,
                        self.num_heads,
                    )
                    + attn_probs.size()[1:]
                ),
                v.view(
                    (
                        kv_bsz,
                        self.num_heads,
                    )
                    + v.size()[1:]
                ),
            )
            attn = attn.reshape((-1,) + attn.size()[-2:])
        else:
            attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights



@register_model_architecture(
    "testdeltalm", "testdeltalm_base"
)
def base_architecture(args):
    args.encoder_embed_dim = 768
    args.encoder_ffn_embed_dim = 3072
    args.encoder_layers = 12
    args.encoder_attention_heads = 12
    args.encoder_normalize_before = False
    args.encoder_learned_pos = True
    args.decoder_embed_dim = 768
    args.decoder_ffn_embed_dim = 3072
    args.decoder_layers = 6
    args.decoder_attention_heads = 12
    args.decoder_normalize_before = False
    args.decoder_learned_pos = True
    args.activation_fn = "gelu"
    args.no_scale_embedding = True
    args.layernorm_embedding = True
    args.max_positions = 512


@register_model_architecture(
    "testdeltalm", "testdeltalm_large"
)
def large_architecture(args):
    base_architecture(args)
    args.encoder_embed_dim = 1024
    args.encoder_ffn_embed_dim = 4096
    args.encoder_layers = 24
    args.encoder_attention_heads = 16
    args.encoder_normalize_before = False
    args.decoder_embed_dim = 1024
    args.decoder_ffn_embed_dim = 4096
    args.decoder_layers = 12
    args.decoder_attention_heads = 16
    args.decoder_normalize_before = False
    args.layernorm_embedding = False
