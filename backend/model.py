import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out, attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention_out, attention_matrix = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention_out + query))
        forward_out = self.feed_forward(x)
        out = self.dropout(self.norm2(forward_out + x))
        
        return out, attention_matrix

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length,
        embed_size=64,
        num_layers=1, 
        heads=2,
        forward_expansion=4,
        dropout=0.1,
    ):
        super(MiniGPT, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x, mask=None, return_states=False):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        
        word_embeds = self.word_embedding(x)
        pos_embeds = self.position_embedding(positions)
        out = self.dropout(word_embeds + pos_embeds)

        states = {
            "tokens": x,
            "word_embeddings": word_embeds,
            "position_embeddings": pos_embeds,
            "combined_embeddings": out,
            "attention_matrices": []
        }

        if mask is None:
            mask = torch.tril(torch.ones((seq_length, seq_length))).expand(
                N, 1, seq_length, seq_length
            ).to(x.device)

        for layer in self.layers:
            out, att_matrix = layer(out, out, out, mask)
            if return_states:
                states["attention_matrices"].append(att_matrix)

        logits = self.fc_out(out)
        
        if return_states:
            return logits, states
        return logits