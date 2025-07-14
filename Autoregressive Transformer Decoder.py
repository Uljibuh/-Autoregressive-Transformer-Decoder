import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# AR Transformer Decoder (same as before)
class ARTransformerDecoder(nn.Module):
    def __init__(self, gene_dim, latent_dim, num_heads=4, num_layers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        
        self.gene_dim = gene_dim
        self.latent_dim = latent_dim
        
        # Gene value embedding
        self.gene_embedding = nn.Linear(1, ff_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, gene_dim, ff_dim))
        
        # Project latent z to decoder embedding dimension
        self.latent_proj = nn.Linear(latent_dim, ff_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final output projection to gene value
        self.output_layer = nn.Linear(ff_dim, 1)
        
    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, gene_values, latent_z):
        batch_size, gene_dim, _ = gene_values.shape
        
        gene_emb = self.gene_embedding(gene_values)
        gene_emb = gene_emb + self.pos_embedding[:, :gene_dim, :]
        
        latent_proj = self.latent_proj(latent_z).unsqueeze(1)
        latent_proj = latent_proj.repeat(1, gene_dim, 1)
        
        decoder_input = gene_emb + latent_proj
        decoder_input = decoder_input.permute(1, 0, 2)
        
        causal_mask = self.generate_causal_mask(gene_dim).to(gene_values.device)
        
        output = self.transformer_decoder(tgt=decoder_input, memory=None, tgt_mask=causal_mask)
        output = self.output_layer(output)
        output = output.permute(1, 0, 2)
        
        return output
