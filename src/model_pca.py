"""
Module that provides a classifier to train a model on embeddings and predict
whether a protein is ARG or not. The dataset used is the antibiotic-resistance
from biodatasets, and the embedding of the 17k proteins come from the 
esm1_t34_670M_UR100 model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from biodatasets import list_datasets, load_dataset
from deepchain.models import MLP
from deepchain.models.utils import (confusion_matrix_plot,
                                    model_evaluation_accuracy)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import Tensor, float32, float64, norm, randperm
from torchmetrics import F1, Accuracy, AveragePrecision, Precision, Recall


class ModelPCA(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Load embedding and target dataset
        dataset = pd.read_csv(
            "/home/selim/Documents/myApps/Antibiotic-resistanceTest/data/dataset.csv"
        )

        y = np.array(list(dataset["label"]))

        cls_embeddings = np.load(
            "/home/selim/Documents/myApps/Antibiotic-resistanceTest/data/sequence_esm1_t34_670M_UR100_cls_embeddings.npy",
            allow_pickle=True,
        )

        cls_embeddings = torch.tensor(np.vstack(cls_embeddings).astype(np.float))
        normalized = StandardScaler().fit_transform(cls_embeddings)
        self.pca = PCA(n_components=0.95)
        self.pca = self.pca.fit(normalized)

        

    
   








