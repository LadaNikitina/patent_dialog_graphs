import torch
import numpy as np
import dgl
import pickle
from sklearn.preprocessing import OneHotEncoder

class PersonaEmbeddings:
    def __init__(self, GRAPH_PATH):
        with open(GRAPH_PATH, 'rb') as fp:
            self.clusters = pickle.load(fp)

        self.one_hot_encoder = OneHotEncoder().fit(np.arange(self.clusters.second_num_clusters).reshape(-1, 1))

    
    def create_representation(self, dialog):
        persona_utterances_embs = []
        
        utterance_clusters = [self.clusters.utterance_cluster_search(utterance, speaker)[0]
                              for utterance, speaker in zip(dialog["utterance"], dialog["speaker"])]
                    
        return np.sum(self.one_hot_encoder.transform(np.array(utterance_clusters).reshape(-1, 1)).toarray(), axis=0)
