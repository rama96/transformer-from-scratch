""" Contains the whoel transformers Architecture - Encoder , Decoder and combined """
import torch.nn as nn
from models.utils import Embeddings
from models.utils import MultiAttentionHead , FeedForward

class TransformerEndcoderLayer:
    """ Contains the build blocks for an Encoder Transformer """
    def __init__(self,config) -> None:
        
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.MultiAttentionHead = MultiAttentionHead(config)
        self.FeedForward = FeedForward(config)

    def forward(self , x):
        
        x = x + self.MultiAttentionHead(self.layer_norm_1(x))
        x = x + self.FeedForward(self.layer_norm_2(x))
        return x
        

class TransformerEncoder:
    """ Encoder Block of a transformer  """
    def __init__(self,config) -> None:
        self.Embeddings = Embeddings(config)
        self.encoder_layers = nn.ModuleList([TransformerEndcoderLayer(config) for _ in config.n_hidden_layers])

    def forward(self , x):
        x = self.Embeddings(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class TransformerDecoderLayer:
    """ Contains the build blocks for an Decoder Transformer """
    def __init__(self) -> None:
        pass
    def forward(self , x):
        pass

class TransformerDecoder:
    """ Contains the build blocks for an Decoder Transformer """
    def __init__(self) -> None:
        pass
    def forward(self , x):
        pass
