""" Contains the whole transformers Architecture - Encoder , Decoder and combined along with the basic building blocks required to create it """
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Embeddings(nn.Module):
    
    """ A class that is used to encode inputs/words into tensors which is later fed into Encoder/Decoder layers.
    Contains 4 main steps in the forward method
    1. Token Embeddings - Embedddings for tokens
    2. Positional Embeddings - Embeddings for positions 
    3. LayerNorm - Normalizing agent
    4. Dropout - A commonly used regulrization method in NN
    """
    
    def __init__(self,config):
        super().__init__()
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


def scalar_dot_product(q,k,v,masked=None):
    """ Accepts 3 args namely :     
    1. q - Query 
    2. k - Key 
    3. v - Value 

    All these three components are generated by passing the input IDs through 3 independent Neural Networks .
    Output is generated using the following formula out = softmax(Q @ K.T) @ V

    """
    dim_k = k.size(-1)
    scores = torch.bmm(q,k.transpose(1,2)) / sqrt(dim_k)
    
    # TODO : Masked Attention implementation for the decoder
    if masked is not None :
        seq_len = scores.size(1)
        mask = torch.trill(torch.ones(seq_len, seq_len)).unsqueeze(0)
        scores.masked_fill(mask==0,value = -float('inf'))

    weights = F.softmax(scores,dim=-1)
    return torch.bmm(weights,v)



class AttentionHead(nn.Module):
    """ Splits the inputs into query , key and value and then performs scalar dot product to get the output for the same 
    """
    def __init__(self,config):
        super().__init__()
        head_dim = int((config.hidden_size / config.num_attention_heads))
        self.linear_k = nn.Linear(config.hidden_size , head_dim)
        self.linear_v = nn.Linear(config.hidden_size , head_dim)
        self.linear_q = nn.Linear(config.hidden_size , head_dim)


    def forward(self,x):
        return scalar_dot_product(
            self.linear_q(x) , self.linear_k(x) , self.linear_v(x)
        )

class MultiAttentionHead(nn.Module):
    """ Creates n_head Attention Heads , concats the outputs , runs them into a linear NN to get the output
    Analgolus to CausalSelfAttention in miniGPT written by Andrej Karpathy :
    https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py
    """
    def __init__(self,config):
        super().__init__()
        self.output_linear = nn.Linear(config.hidden_size , config.hidden_size)
        self.attention_heads = nn.ModuleList([AttentionHead(config) for _ in range(config.num_attention_heads)])
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.attention_heads] , dim=-1)
        return self.output_linear(out)


class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cfc = nn.Linear(config.hidden_size , config.intermediate_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear( config.intermediate_size , config.hidden_size)
        self.dropout = nn.Dropout()
    
    def forward(self,x):
        x = self.linear_1(x) 
        x = self.gelu(x) 
        x = self.linear_2(x) 
        x = self.dropout(x) 
        return x

class TransformerEndcoderLayer(nn.Module):
    """ Contains the build blocks for an Encoder Transformer 
    Analgolus to Block in miniGPT written by Andrej Karpathy :
    https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py
    """
    def __init__(self,config) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.MultiAttentionHead = MultiAttentionHead(config)
        self.FeedForward = FeedForward(config)

    def forward(self , x):
        
        x = x + self.MultiAttentionHead(self.layer_norm_1(x))
        x = x + self.FeedForward(self.layer_norm_2(x))
        return x

# class TransformerDecoderLayer(nn.Module):
#     """ Contains the build blocks for an Encoder Transformer 
#     Analgolus to Block in miniGPT written by Andrej Karpathy :
#     https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py
#     """
#     def __init__(self,config) -> None:
#         super().__init__()
#         self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
#         self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
#         self.MultiAttentionHead = MultiAttentionHead(config)
#         self.FeedForward = FeedForward(config)

#     def forward(self , x):
        
#         x = x + self.MultiAttentionHead(self.layer_norm_1(x))
#         x = x + self.FeedForward(self.layer_norm_2(x))
#         return x



class Encoder:
    """ Encoder Block of a transformer  """
    def __init__(self,config) -> None:
        super().__init__()
        self.Embeddings = Embeddings(config)
        self.encoder_layers = nn.ModuleList([TransformerEndcoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self , x):
        x = self.Embeddings(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class TransformersForClassification(nn.Module):
    """ Classifies the given text as positive or negative or neutral """
    def __init__(self , config):
        super().__init__()
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        
    def forward(self,x):
        x = self.encoder(x)[: , 0 , :]
        x = self.dropout(x)
        x = self.classifier(x)
        return x




# class Decoder:
#     """ Encoder Block of a transformer  """
#     def __init__(self,config) -> None:
#         super().__init__()
#         self.Embeddings = Embeddings(config)
#         self.encoder_layers = nn.ModuleList([TransformerEndcoderLayer(config) for _ in range(config.num_hidden_layers)])

#     def forward(self , x):
#         x = self.Embeddings(x)
#         for layer in self.encoder_layers:
#             x = layer(x)
#         return x


if __name__ == "__main__":
    encoder = Encoder()
    text = "This pizza was really good"

