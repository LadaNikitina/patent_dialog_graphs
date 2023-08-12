import torch
import numpy as np
import dgl
import pickle
from torch.utils.data import DataLoader
from scipy.spatial import distance

class Ranking:
    def __init__(self, GRAPH_PATH, MODEL_PATHS):
        if len(MODEL_PATHS) == 1:
            self.num_speakers = 1
            
            self.GAT_model = torch.load(MODEL_PATHS[0])
            self.top_k = self.GAT_model.top_k

        elif len(MODEL_PATHS) == 2:
            self.num_speakers = 2

            self.user_GAT_model = torch.load(MODEL_PATHS[0])
            self.system_GAT_model = torch.load(MODEL_PATHS[1])
            self.top_k = self.user_GAT_model.top_k

        
        with open(GRAPH_PATH, 'rb') as fp:
            self.clusters = pickle.load(fp)
    
    def ranking(self, dialog, next_utterances):
        # 0 - user, 1 - system
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        graph = []
        embs = []
        
        speakers = dialog["speaker"]
        utterances = dialog["utterance"]
        
        if len(dialog["speaker"]) < self.top_k:
            for _ in range(self.top_k - len(dialog["utterance"])):
                graph.append(self.clusters.second_num_clusters)
                embs.append(np.zeros(self.clusters.embs_dim))
        else:
            speakers = dialog["speaker"][( - self.top_k) : ]
            utterances = dialog["utterance"][( - self.top_k) : ]

        for utterance, speaker in zip(utterances, speakers):
            cluster, embedding = self.clusters.utterance_cluster_search(utterance, speaker)
            
            graph.append(cluster)
            embs.append(embedding)

        edges = list(range(self.top_k))
        g = dgl.graph((edges[:-1], edges[1:]), num_nodes = self.top_k)
        
        g.ndata['attr'] = torch.tensor(graph, dtype=torch.int64)
        g.ndata['emb'] = torch.tensor(np.array(embs))
        
        g = dgl.add_self_loop(g) 
        
        if self.num_speakers == 2:
            if dialog["speaker"][-1] == 0:
                model = self.system_GAT_model.to(device)
            else:
                model = self.user_GAT_model.to(device)
        else:
            model = self.GAT_model.to(device)
        
        model.eval()
        result_cluster = np.argmax(torch.softmax(model(g.to(device)), 1).tolist()[0])

        ranking_result = []

        if self.num_speakers == 2:
            if dialog["speaker"][-1] == 0:
                cluster_center = self.clusters.system_mean_emb[result_cluster]
            else:
                cluster_center = self.clusters.user_mean_emb[result_cluster]
        else:
            cluster_center = self.clusters.mean_emb[result_cluster]
        
        for next_utterance in next_utterances:
            utterance_embedding = self.clusters.encoder_model.encode(next_utterance)
            ranking_result.append((next_utterance, distance.euclidean(utterance_embedding, cluster_center)))
            
        return sorted(ranking_result, key = lambda x: x[1])