from collections import Counter
import pathlib
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import torch
import numpy as np
import json
import itertools
import pandas as pd
import faiss
import random
import os

def get_pd_utterances_speaker(data):
    '''
        parsing data
    '''
    utterances = []

    for obj in data:
        utterances += obj['utterance']

    speakers = []

    for obj in data:
        speakers += obj['speaker']    
    
    df = pd.DataFrame()
    
    df['utterance'] = utterances
    df['speaker'] = speakers
    
    return df


class Clusters:
    ''' 
        the class that forms clusters
    '''
    def __init__(self, data_path, embedding_file, language, first_num_clusters, second_num_clusters):
        self.data_path = data_path
        self.embedding_file = embedding_file
        self.first_num_clusters = first_num_clusters
        self.second_num_clusters = second_num_clusters
        self.language = language
        
        if self.first_num_clusters == -1:
            self.first_num_clusters = self.second_num_clusters
            self.second_num_clusters = -1
            
    def data_loading(self):
        '''
            data loading
        '''
        
        with open(self.data_path) as file:
            dataset = json.load(file)
        
        random.shuffle(dataset)
        # train-validation splitting 
        validation_split = int(len(dataset) * 0.8)
        self.valid_dataset = dataset[validation_split : ]
        self.train_dataset = dataset[ : validation_split]
        
        # get utterances from data
        train_df = get_pd_utterances_speaker(self.train_dataset)
        valid_df = get_pd_utterances_speaker(self.valid_dataset)
        
        self.df = pd.concat([train_df, valid_df], ignore_index=True)

        # split data on user/system train/valid
        self.user_train_df = train_df[train_df["speaker"] == 0].reset_index(drop=True)
        self.user_valid_df = valid_df[valid_df["speaker"] == 0].reset_index(drop=True)
        
        self.system_train_df = train_df[train_df["speaker"] == 1].reset_index(drop=True)
        self.system_valid_df = valid_df[valid_df["speaker"] == 1].reset_index(drop=True)
        
        # get user/system train/valid indexes for getting embeddings
        self.user_train_index = train_df[train_df["speaker"] == 0].index
        self.user_valid_index = valid_df[valid_df["speaker"] == 0].index + len(train_df)
        
        self.system_train_index = train_df[train_df["speaker"] == 1].index
        self.system_valid_index = valid_df[valid_df["speaker"] == 1].index + len(train_df)

    def get_embeddings(self):
        '''
            calculating embeddings
        '''
        
        if self.language == "ru":
            self.encoder_model = SentenceTransformer('sentence-transformers/LaBSE')
        elif self.language == "en":
            self.encoder_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        else:
            raise ValueError('Wrong language!')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_model = self.encoder_model.to(device)
        if pathlib.Path(self.embedding_file).exists():
            embeddings = np.load(self.embedding_file)
        else:
            embeddings = self.encoder_model.encode(self.df["utterance"])
            np.save(self.embedding_file, embeddings)
        
        self.embs_dim = embeddings.shape[1]
        # train user/system embeddings
        self.train_user_embs = embeddings[self.user_train_index]
        self.train_system_embs = embeddings[self.system_train_index]

        # validation user/system embeddings
        self.valid_user_embs = embeddings[self.user_valid_index]
        self.valid_system_embs = embeddings[self.system_valid_index]
        
    
    def get_first_clusters(self, embs, n_clusters):
        '''
            first-stage clustering
        '''
        kmeans = faiss.Kmeans(embs.shape[1], n_clusters, verbose = True, max_points_per_centroid = 5000)
        kmeans.train(embs.astype(np.float32, copy=False))

        _, labels = kmeans.index.search(embs.astype(np.float32, copy=False), 1)
        return labels.squeeze()
    
    def first_stage(self):
        '''
            creating first-stage clusters
        '''
        self.train_user_df_first_stage = self.user_train_df.copy()
        self.train_system_df_first_stage = self.system_train_df.copy()

        self.train_user_df_first_stage['cluster'] = self.get_first_clusters(self.train_user_embs, self.first_num_clusters)
        self.train_system_df_first_stage['cluster'] = self.get_first_clusters(self.train_system_embs, self.first_num_clusters)

        # counting center of mass of the cluster
        self.user_mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))
        self.system_mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))

        for i in range(self.first_num_clusters):
            index_cluster = self.train_user_df_first_stage[self.train_user_df_first_stage['cluster'] == i].index
            self.user_mean_emb[i] = np.mean(self.train_user_embs[index_cluster], axis = 0)

            index_cluster = self.train_system_df_first_stage[self.train_system_df_first_stage['cluster'] == i].index
            self.system_mean_emb[i] = np.mean(self.train_system_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        ind_user = 0
        ind_system = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    utterance_clusters.append(str(self.train_user_df_first_stage["cluster"][ind_user]) + "-user")
                    ind_user += 1
                else:
                    utterance_clusters.append(str(self.train_system_df_first_stage["cluster"][ind_system]) + "-system")
                    ind_system += 1

            array_for_word2vec.append(utterance_clusters)       

        model_first_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.user_cluster_embs_first_stage = []
        self.system_cluster_embs_first_stage = []

        for i in range(self.first_num_clusters):
            self.user_cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)  + "-user"]))
            self.system_cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)  + "-system"]))
    
    def get_validation_clusters(self, num_clusters):
        '''
            cluster searching for validation
        '''

        self.valid_user_df = self.user_valid_df.copy()
        self.valid_system_df = self.system_valid_df.copy()

        # searching the nearest cluster for each validation user utterance
        valid_user_clusters = []

        for i in range(len(self.valid_user_df)):
            distances = []
            emb = np.array(self.valid_user_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.user_mean_emb[j]))), j))

            distances = sorted(distances)
            valid_user_clusters.append(distances[0][1])

        self.valid_user_df['cluster'] = valid_user_clusters      

        # searching the nearest cluster for each validation system utterance
        valid_system_clusters = []

        for i in range(len(self.valid_system_df)):
            distances = []
            vec = np.array(self.valid_system_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(vec - self.system_mean_emb[j]))), j))

            distances = sorted(distances)
            valid_system_clusters.append(distances[0][1])

        self.valid_system_df['cluster'] = valid_system_clusters
        
        
    def utterance_cluster_search(self, utterance, speaker):
        distances = []
        utterance_embedding = self.encoder_model.encode(utterance)
        
        if speaker == 0:
            for j in range(self.second_num_clusters):
                distances.append((np.sqrt(np.sum(np.square(utterance_embedding - self.user_mean_emb[j]))), j))
        elif speaker == 1:
            for j in range(self.second_num_clusters):
                distances.append((np.sqrt(np.sum(np.square(utterance_embedding - self.system_mean_emb[j]))), j))

        distances = sorted(distances)
        return distances[0][1], utterance_embedding
        
            
    def one_stage_clustering(self):
        '''
            one stage clustering
        '''
 
        self.get_validation_clusters(self.first_num_clusters)
        self.user_cluster_embs = self.user_cluster_embs_first_stage
        self.system_cluster_embs = self.system_cluster_embs_first_stage
        self.train_user_df = self.train_user_df_first_stage
        self.train_system_df = self.train_system_df_first_stage
          
    def second_stage(self):
        '''
            creating second-stage clusters
        '''
        # creating user second-stage clusters
        self.train_user_df_sec_stage = self.user_train_df.copy()

        kmeans = faiss.Kmeans(len(self.user_cluster_embs_first_stage[0]), self.second_num_clusters, min_points_per_centroid=10)
        kmeans.train(np.array(self.user_cluster_embs_first_stage).astype(np.float32, copy=False))
    
        _, user_new_clusters = kmeans.index.search(np.array(self.user_cluster_embs_first_stage).astype(np.float32, copy=False), 1)
        user_new_clusters = user_new_clusters.squeeze()

        new_user_clusters = []

        for i in range(len(self.train_user_df_first_stage)):
            cur_cluster = self.train_user_df_first_stage['cluster'][i]
            new_user_clusters.append(user_new_clusters[cur_cluster])

        self.train_user_df_sec_stage['cluster'] = new_user_clusters
        
        # creating system second-stage clusters
        self.train_system_df_sec_stage = self.system_train_df.copy()

        kmeans = faiss.Kmeans(len(self.system_cluster_embs_first_stage[0]), self.second_num_clusters, min_points_per_centroid=10)
        kmeans.train(np.array(self.system_cluster_embs_first_stage).astype(np.float32, copy=False))
    
        _, system_new_clusters = kmeans.index.search(np.array(self.system_cluster_embs_first_stage).astype(np.float32, copy=False), 1)
        system_new_clusters = system_new_clusters.squeeze()

        new_sys_clusters = []

        for i in range(len(self.train_system_df_first_stage)):
            cur_cluster = self.train_system_df_first_stage['cluster'][i]
            new_sys_clusters.append(system_new_clusters[cur_cluster])

        self.train_system_df_sec_stage['cluster'] = new_sys_clusters

        # counting center of mass of the cluster        
        self.user_mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))
        self.system_mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))

        for i in range(self.second_num_clusters):
            index_cluster = self.train_user_df_sec_stage[self.train_user_df_sec_stage['cluster'] == i].index
            self.user_mean_emb[i] = np.mean(self.train_user_embs[index_cluster], axis = 0)

            index_cluster = self.train_system_df_sec_stage[self.train_system_df_sec_stage['cluster'] == i].index
            self.system_mean_emb[i] = np.mean(self.train_system_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        ind_user = 0
        ind_system = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    utterance_clusters.append(str(self.train_user_df_sec_stage["cluster"][ind_user]) + "-user")
                    ind_user += 1
                else:
                    utterance_clusters.append(str(self.train_system_df_sec_stage["cluster"][ind_system]) + "-system")
                    ind_system += 1

            array_for_word2vec.append(utterance_clusters)       

        model_sec_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.user_cluster_embs_sec_stage = []
        self.system_cluster_embs_sec_stage = []

        for i in range(self.second_num_clusters):
            self.user_cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)  + "-user"]) + list(self.user_mean_emb[i]))
            self.system_cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)  + "-user"]) + list(self.system_mean_emb[i]))
    
    def two_stage_clustering(self):
        '''
            two_stage_clustering
        '''

        self.get_validation_clusters(self.second_num_clusters)
        self.user_cluster_embs = self.user_cluster_embs_sec_stage
        self.system_cluster_embs = self.system_cluster_embs_sec_stage
        self.train_user_df = self.train_user_df_sec_stage
        self.train_system_df = self.train_system_df_sec_stage
        
    def form_clusters(self):
        '''
            formation of clusters
        '''
        print("The data is loading...")
        self.data_loading()
        print("The embeddings are loading...")
        self.get_embeddings()
        print("The first stage of clustering has begun...")
        self.first_stage()
        
        if self.second_num_clusters == -1:
            self.one_stage_clustering()
        else:
            print("The second stage of clustering has begun...")
            self.second_stage()
            print("The searching clusters for validation has begun...")
            self.two_stage_clustering()