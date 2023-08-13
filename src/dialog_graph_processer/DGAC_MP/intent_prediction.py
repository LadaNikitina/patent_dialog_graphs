import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dialog_graph_processer.DGAC_MP.dgac_two_speakers import Clusters as ClustersTwoSpeakers
from dialog_graph_processer.DGAC_MP.dgac_one_speakers import Clusters as ClustersOneSpeaker
from dialog_graph_processer.DGAC_MP.data_function_one_partite import get_data as get_data_one_speaker
from dialog_graph_processer.DGAC_MP.data_function_two_partite import get_data as get_data_two_speakers
from dialog_graph_processer.DGAC_MP.dgl_graph_conctruction import get_dgl_graphs
from dialog_graph_processer.DGAC_MP.early_stopping_tools import LRScheduler, EarlyStopping
from dialog_graph_processer.DGAC_MP.GAT import GAT_model


class IntentPredictor:
    def __init__(self, data_path, embedding_file, language, num_speakers, num_clusters_per_stage):
        self.data_path = data_path
        self.embedding_file = embedding_file
        self.language = language
        self.num_speakers = num_speakers

        if len(num_clusters_per_stage) > 2 or len(num_clusters_per_stage) == 0:
            raise ValueError("Wrong number of clustering stages")

        self.second_stage_num_clusters = num_clusters_per_stage[-1]

        if len(num_clusters_per_stage) == 1:
            first_stage_num_clusters = -1
        else:
            first_stage_num_clusters = num_clusters_per_stage[0]

        if self.num_speakers > 2 or self.num_speakers < 1:
            raise ValueError("Wrong number of speakers")
        elif self.num_speakers == 1:
            self.clusters = ClustersOneSpeaker(
                self.data_path, self.embedding_file, language, first_stage_num_clusters, self.second_stage_num_clusters
            )
        else:
            self.clusters = ClustersTwoSpeakers(
                self.data_path, self.embedding_file, language, first_stage_num_clusters, self.second_stage_num_clusters
            )

    def dialog_graph_auto_construction(self):
        self.clusters.form_clusters()

    def dump_dialog_graph(self, PATH):
        pickle.dump(self.clusters, open(PATH, "wb"))

    def one_partite_dgl_graphs_preprocessing(self):
        train_x, train_y = get_data_one_speaker(
            self.clusters.train_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.cluster_train_df,
            np.array(self.clusters.train_embs.astype(np.float64, copy=False)),
        )

        valid_x, valid_y = get_data_one_speaker(
            self.clusters.valid_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.cluster_valid_df,
            np.array(self.clusters.valid_embs.astype(np.float64, copy=False)),
        )

        self.train_dataloader = get_dgl_graphs(train_x, train_y, self.top_k, self.batch_size)
        self.valid_dataloader = get_dgl_graphs(valid_x, valid_y, self.top_k, self.batch_size)

    def two_partite_dgl_graphs_preprocessing(self):
        user_train_x, user_train_y, sys_train_x, sys_train_y = get_data_two_speakers(
            self.clusters.train_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.train_user_df,
            self.clusters.train_system_df,
            np.array(self.clusters.train_user_embs).astype(np.float64, copy=False),
            np.array(self.clusters.train_system_embs).astype(np.float64, copy=False),
        )

        user_valid_x, user_valid_y, sys_valid_x, sys_valid_y = get_data_two_speakers(
            self.clusters.valid_dataset,
            self.top_k,
            self.second_stage_num_clusters,
            self.clusters.valid_user_df,
            self.clusters.valid_system_df,
            np.array(self.clusters.valid_user_embs).astype(np.float64, copy=False),
            np.array(self.clusters.valid_system_embs).astype(np.float64, copy=False),
        )

        self.user_train_dataloader = get_dgl_graphs(user_train_x, user_train_y, self.top_k, self.batch_size)
        self.sys_train_dataloader = get_dgl_graphs(sys_train_x, sys_train_y, self.top_k, self.batch_size)

        self.user_valid_dataloader = get_dgl_graphs(user_valid_x, user_valid_y, self.top_k, self.batch_size)
        self.sys_valid_dataloader = get_dgl_graphs(sys_valid_x, sys_valid_y, self.top_k, self.batch_size)

    def dgl_graphs_preprocessing(self, top_k=10, batch_size=128):
        self.top_k = top_k
        self.batch_size = batch_size

        if self.num_speakers == 1:
            self.one_partite_dgl_graphs_preprocessing()
        else:
            self.two_partite_dgl_graphs_preprocessing()

    def init_message_passing_model(self, PATH, num_heads=2, hidden_dim=512):
        embs_dim = self.clusters.embs_dim

        if self.num_speakers == 1:
            self.train_model(self.train_dataloader, self.valid_dataloader, PATH + "/GAT")
        else:
            self.train_model(self.user_train_dataloader, self.user_valid_dataloader, PATH + "/GAT_user")
            self.train_model(self.sys_train_dataloader, self.sys_valid_dataloader, PATH + "/GAT_system")

    def train_model(
        self,
        train_dataloader,
        valid_dataloader,
        model_file_name,
        early_stopping_steps=5,
        hidden_dim=512,
        num_heads=2,
        learning_rate=0.0001,
        num_epochs=100,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GAT_model(self.clusters.embs_dim, hidden_dim, num_heads, self.top_k, self.second_stage_num_clusters).to(
            device
        )

        for param in model.parameters():
            param.requires_grad = True

        train_loss_values = []
        valid_loss_values = []

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = LRScheduler(optimizer)
        early_stopping = EarlyStopping(early_stopping_steps)

        for epoch in range(num_epochs):
            train_epoch_loss = 0

            for iter, (batched_graph, labels) in tqdm(enumerate(train_dataloader)):
                logits = model(batched_graph.to(device))
                loss = criterion(logits, labels.to(device))
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.detach().item()

            train_epoch_loss /= iter + 1
            train_loss_values.append(train_epoch_loss)

            valid_epoch_loss = 0
            with torch.no_grad():
                for iter, (batched_graph, labels) in enumerate(valid_dataloader):
                    logits = model(batched_graph.to(device))
                    loss = criterion(logits, labels.to(device))
                    valid_epoch_loss += loss.detach().item()

                valid_epoch_loss /= iter + 1
                valid_loss_values.append(valid_epoch_loss)

            print(f"Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}")

            lr_scheduler(valid_epoch_loss)
            early_stopping(valid_epoch_loss)

            if early_stopping.early_stop:
                break

        torch.save(model, model_file_name)
