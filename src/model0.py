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


class Model0(torch.nn.Module):

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
    pca = PCA(n_components=0.95)
    pca = pca.fit(normalized)
    lowdim_embeddings = pca.transform(normalized)

    x_tsne = TSNE().fit_transform(lowdim_embeddings)
    df = pd.DataFrame()
    df["tsne-2d-one"] = x_tsne[:,0]
    df["tsne-2d-two"] = x_tsne[:,1]
    df["target"] = y

    plt.figure(figsize=(16,10))

    sns.scatterplot(
        data = df,
        x="tsne-2d-one", y="tsne-2d-two",
        hue ="target",
        palette=sns.color_palette("hls",2),
        legend="full",
        alpha =0.3
    )
    plt.savefig('tsne.png')


    #159 components, 95% variance kepts

    x_train, x_test, y_train, y_test = train_test_split(np.array(lowdim_embeddings), y, test_size=0.3)


    # Build a multi-layer-perceptron on top of embedding

    # The fit method can handle all the arguments available in the
    # 'trainer' class of pytorch lightening :
    #               https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    # Example arguments:
    # * specifies all GPUs regardless of its availability :
    #               Trainer(gpus=-1, auto_select_gpus=False, max_epochs=20)

    #---------------------------------Logistic Regression-------------------------------------------------

    #logreg = LogisticRegression(random_state=0, solver='liblinear').fit(x_train, y_train)
    #y_pred = logreg.predict_proba(x_test)
    #y_pred = np.array([x[1] for x in y_pred])


    #-----------------------------------------------------------------------------------------------------

    #---------------------------------SVM-------------------------------------------------

    #svm = SVC(probability=True).fit(x_train, y_train)
    #y_pred = svm.predict_proba(x_test)
    #y_pred = np.array([x[1] for x in y_pred])


    #-----------------------------------------------------------------------------------------------------



    #---------------------------------MLP-----------------------------------------------------------------

    n_class = len(np.unique(y_train))
    #print(n_class)
    input_shape = x_train.shape[1]
    #print(input_shape)

    mlp = MLP(input_shape=input_shape, n_class=n_class)
    mlp.fit(x_train, y_train, epochs=16)
    mlp.save("model.pt")

    # Model evaluation
    y_pred = mlp(x_test).squeeze().detach().numpy()
    model_evaluation_accuracy(y_test, y_pred)

    #------------------------------------------------------------------------------------------------------

    # Plot confusion matrix
    confusion_matrix_plot(y_test, (y_pred > 0.5).astype(int), ["0", "1"])

    # Accuracy evaluation
    accuracy = Accuracy()
    print('Accuracy: {0}'.format(accuracy(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))

    # Precision evaluation
    precision = Precision()
    print('Precision: {0}'.format(precision(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))

    # Recall evalution
    recall = Recall()
    print('Recall: {0}'.format(recall(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))

    # F1 Score evaluation
    f1score = F1()
    print('F1 Score: {0}'.format(f1score(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))








