from dgl import nn as dgl_nn
from torch import nn

import numpy as np
import torch

class GAT_model(nn.Module):
    def __init__(self, embs_dim, hidden_dim, num_heads, top_k, num_classes):
        super(GAT_model, self).__init__()

        self.top_k = top_k
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.embs_dim = embs_dim
        self.hidden_dim = hidden_dim
        
        self.layer1 = dgl_nn.GATv2Conv(self.embs_dim, self.hidden_dim, self.num_heads)
        self.layer2 = dgl_nn.GATv2Conv(self.hidden_dim * self.num_heads, self.hidden_dim, self.num_heads)

        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.2)

        linear_weights = np.zeros(self.top_k)
        linear_weights[...] = 1 / self.top_k
        linear_weights = torch.tensor(linear_weights).view(1, -1)

        self.positional_encoding = nn.Embedding.from_pretrained(linear_weights.float()).requires_grad_(True)  
        self.classify = nn.Linear(self.hidden_dim * self.num_heads, self.num_classes)

    def forward(self, graph):
        x_emb = graph.ndata['emb']

        h = x_emb.to(torch.float32)

        h = self.layer1(graph, h)
        h = self.do1(h)
        h = torch.reshape(h, (h.shape[0], self.num_heads * self.hidden_dim))      
        h = self.layer2(graph, h)
        h = self.do2(h)


        graph.ndata['h'] = h

        h = torch.reshape(h, (h.shape[0] // self.top_k, self.top_k, self.num_heads * self.hidden_dim))        
        linear_weights = torch.reshape(self.positional_encoding.weight, (self.top_k, ))

        get_pos_encoding = lambda embedding: torch.matmul(linear_weights, embedding)
        h = list(map(get_pos_encoding, h))
        h = torch.stack(h)
        return self.classify(h)  