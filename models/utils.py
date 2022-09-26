import torch
import torch.nn as nn
## Scalar Dot Product


class Embeddings(nn.Module):
    
    """ A class that is used to encode inputs/words into tensors which is later fed into Encoder/Decoder layers.
    Contains 4 main steps in the forward method
    1. Token Embeddings - Embedddings for tokens
    2. Positional Embeddings - Embeddings for positions 
    3. LayerNorm - Normalizing agent
    4. Dropout - A commonly used regulrization method in NN
    """
    
    def __init__(self,config):
        self.token_embeddings = nn.Embedding(config.vocab_size , config.hidden_size)
        self.positional_embeddings = nn.Embedding(config.max_position_embeddings , config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size , eps=1e-12)
        self.dropout = nn.Dropout()
    
    def forward(self,input_ids):
        
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len , dtype = torch.long ).unsqueeze(0)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.token_embeddings(positions)

        embeddings = token_embeddings + position_embeddings
        
        out = self.dropout(self.LayerNorm(embeddings))
        return out


class ScalarDotProduct:
    def __init__(self,config):
        pass
    def forward(self,x):
        pass

class AttentionHead:
    def __init__(self,config):
        pass
    def forward(self,x):
        pass
class MultiAttentionHead:
    def __init__(self,config):
        pass
    def forward(self,x):
        pass
class FeedForward:
    def __init__(self,config):
        pass
    def forward(self,x):
        pass


