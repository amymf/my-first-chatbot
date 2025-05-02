import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, max_len, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
    def forward(self, x, padding_mask):
        # x shape: (batch_size, max_len, embed_dim)
        seq_len = x.size(1)
        x += self.positional_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return x
     
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, max_len, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
    def generate_square_subsequent_mask(self, size):
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        
    def forward(self, x, encoder_output, padding_mask):
        # x shape: (batch_size, seq_len, embed_dim)
        # encoder_output shape: (batch_size, question_len, embed_dim)
        
        seq_len = x.size(1)
        x += self.positional_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask=self.generate_square_subsequent_mask(seq_len), tgt_key_padding_mask=padding_mask)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, max_len, num_heads, num_layers, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, max_len, num_heads, num_layers)
        self.decoder = TransformerDecoder(embed_dim, max_len, num_heads, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, y, input_padding_mask, response_padding_mask):
        # x shape: (batch_size, question_len, embed_dim) - input_id
        # y shape: (batch_size, answer_len, embed_dim) - response_id
        x = self.embedding(x) # (batch_size, question_len, embed_dim)
        y = self.embedding(y) # (batch_size, answer_len, embed_dim)
        
        encoder_output = self.encoder(x, input_padding_mask)
        decoder_output = self.decoder(y, encoder_output, response_padding_mask)
        result = self.fc_out(decoder_output) # (batch_size, answer_len, vocab_size)
        return result
