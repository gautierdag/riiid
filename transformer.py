import torch
import torch.nn as nn
import math
from linear_attention_transformer.autopadder import Autopadder
from linear_attention_transformer import LinearAttentionTransformer
from local_attention import LocalAttention
import pytorch_lightning as pl
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, sequence_length):
        # returns embeds (sequence_length, 1, d_model)
        return self.pe[:sequence_length, :]


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Decoder without additive step in self-attention
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(CustomTransformerDecoderLayer, self).__init__(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.model_type = "CustomTransformerDecoderLayer"

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def get_custom_decoder(
    d_model=64,
    dim_feedforward=256,
    n_heads=2,
    num_decoder_layers=2,
    dropout=0.0,
    activation="relu",
):
    return nn.TransformerDecoder(
        CustomTransformerDecoderLayer(
            d_model, n_heads, dim_feedforward, dropout=dropout, activation=activation
        ),
        num_decoder_layers,
        nn.LayerNorm(d_model),
    )


class LinearTransformerEncDec(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        max_seq_len=8192,
    ):
        super(LinearTransformerEncDec, self).__init__()
        self.encoder = Autopadder(
            LinearAttentionTransformer(
                dim=d_model,
                heads=nhead,
                depth=num_encoder_layers,
                max_seq_len=max_seq_len,
                n_local_attn_heads=nhead,  # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
                causal=True,  # auto-regressive or not
                ff_dropout=dropout,  # dropout for feedforward
                attn_layer_dropout=dropout,  # dropout right after self-attention layer
                attn_dropout=dropout,  # dropout post-attention
            )
        )
        self.decoder = Autopadder(
            LinearAttentionTransformer(
                dim=d_model,
                heads=nhead,
                depth=num_decoder_layers,
                max_seq_len=max_seq_len,
                n_local_attn_heads=nhead,
                causal=True,
                ff_dropout=dropout,  # dropout for feedforward
                attn_layer_dropout=dropout,  # dropout right after self-attention layer
                attn_dropout=dropout,  # dropout post-attention
            )
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        context = self.encoder(src.permute((1, 0, 2)))
        output = self.decoder(tgt.permute((1, 0, 2)), context=context)
        return output.permute(1, 0, 2)


class TransformerEncDec(pl.LightningModule):
    def __init__(
        self,
        transformer_type="base",
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        dim_feedforward=256,
        activation="relu",
    ):
        super(TransformerEncDec, self).__init__()
        self.transformer_type = transformer_type
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer component
        # if self.transformer_type == "linear":
        #     self.transformer = LinearTransformerEncDec(
        #         d_model=d_model,
        #         nhead=nhead,
        #         num_encoder_layers=num_encoder_layers,
        #         num_decoder_layers=num_decoder_layers,
        #         dropout=dropout,
        #     )
        # else:
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
        )

    def generate_square_subsequent_mask(self, sz):
        return torch.tensor(float("-inf"), device=self.device).expand(sz, sz).triu(1)

    def forward(self, embeded_exercises, embeded_responses):

        # adding positional vector
        sequence_length, _, _ = embeded_responses.shape
        embedded_positions = self.pos_encoder(sequence_length + 1)
        # add shifted position embedding ( start token is first position)
        embeded_responses = embeded_responses + embedded_positions[:-1, :, :]
        embeded_exercises = embeded_exercises + embedded_positions[1:, :, :]

        # mask of shape S x S -> prevents attention looking forward
        top_right_attention_mask = self.generate_square_subsequent_mask(sequence_length)
        output = self.transformer(
            embeded_exercises,
            embeded_responses,
            tgt_mask=top_right_attention_mask,  # (T,T)
            src_mask=top_right_attention_mask,  # (S,S)
        )

        return output

