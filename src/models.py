# aig_qor/models.py

import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Set2Set


class DebuggableDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
        need_weights=False,
        attn_weight_type=None,
        average_attn_weights=True,
        **kwargs
    ):
        self_attn_out, self_attn_w = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        tgt = tgt + self.dropout1(self_attn_out)
        tgt = self.norm1(tgt)

        cross_attn_out, cross_attn_w = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        tgt = tgt + self.dropout2(cross_attn_out)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CombinedModel(nn.Module):
    def __init__(self, gcn_in=3, gcn_h=32, gcn_out=64,
                 vocab_size=11, d_model=32, nhead=4,
                 num_layers=4, dropout=0.1, max_len=21):
        super().__init__()
        self.conv1 = GCNConv(gcn_in, gcn_h)
        self.conv2 = GCNConv(gcn_h, gcn_out)
        self.conv3 = GCNConv(gcn_out, gcn_out)
        self.pool = Set2Set(gcn_out, processing_steps=3)

        self.graph_proj = nn.Linear(2 * gcn_out, d_model)
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        dec_layer = DebuggableDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        self.qor_head = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for c in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(c.lin.weight)
        nn.init.xavier_normal_(self.graph_proj.weight)

    def forward(self, data, recipe_tokens):
        x = self.relu(self.conv1(data.x, data.edge_index))
        x = self.relu(self.conv2(x, data.edge_index))
        x = self.relu(self.conv3(x, data.edge_index))
        h = self.pool(x, data.batch)
        h_graph = self.graph_proj(h)

        B, L = recipe_tokens.size()
        rec = recipe_tokens + 1
        graph_tok = torch.zeros(B, 1, dtype=torch.long, device=rec.device)
        tgt_tokens = torch.cat([graph_tok, rec], dim=1)

        emb = self.token_embedding(tgt_tokens)
        emb = self.pos_encoder(emb)

        memory = h_graph.unsqueeze(1).expand(-1, L + 1, -1)
        mask = self._generate_square_subsequent_mask(L + 1, emb.device)

        dec = self.decoder(emb, memory, tgt_mask=mask)
        return self.qor_head(dec)[:, 1:, :]

    def _generate_square_subsequent_mask(self, sz, device):
        m = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        m = m.float().masked_fill(m == 0, float('-inf')).masked_fill(m == 1, 0.0)
        return m.to(device)


class LSTMRecipeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_tokens = self.embedding
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, recipe_tokens):
        x = self.embedding(recipe_tokens)
        _, (h_n, _) = self.lstm(x)
        h_final = h_n[-1]
        return self.output_proj(h_final)


class RecipeCausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4,
                 num_layers=4, dropout=0.1, max_len=21):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.qor_head = nn.Linear(d_model, 1)

    def forward(self, recipe_tokens, return_hidden=False):
        x = self.token_embedding(recipe_tokens)
        x = self.pos_encoder(x)
        L = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        x2 = self.transformer_encoder(x, mask=mask)
        preds = self.qor_head(x2)
        if return_hidden:
            return preds, x2
        return preds
