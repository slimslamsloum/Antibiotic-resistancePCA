# Antibiotic-resistance Prediction
This app is a base application that computes the probability that a protein is antibiotic-resistant (ARG).
The work is mainly based on the paper [HMD-ARG](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01002-3), which describes a more complex model that provides detailed annotations of ARGs.
We first use a bio-transformers with a 'esm1_t34_670M_UR100' model to compute the CLS embeddings. The dataset used is [antibiotic-resistance](https://github.com/DeepChainBio/bio-datasets/blob/main/datasets/antibiotic-resistance/description.md).

We then apply a PCA to the embeddings with desired percentage of variance kept at 95%, which led to 
159 features (instead of the 1280 features of the embeddings). We then apply a T-SNE visualization to the
low-dimensional data, and notice the data classes are well separable:

![tsne.pgn](https://i.postimg.cc/SN7tY8sK/tsne.png)

The default model included is a MLP with input size 1280 (size of an embedding), two hidden dense layers, output size of 1 (binary classification into ARG or not-ARG), and ReLU activation. The model can be described by the following sketch:

![model.png](https://i.postimg.cc/tC0ZWYTZ/Screenshot-from-2022-07-20-13-29-22.png)

The model achieved 0.982 accuracy, 0.974 precision, 0.990 recall and 0.982 F1 Score on the Test set (with 30% of dataset in Test set and 70% in Training set).

The confusion matrix we obtained can be seen here:

![confusion-matrix.png](https://i.postimg.cc/85rWcpkz/confusion-matrix.png)

We notice our model did very well to separate the classes here.

## libraries
- pytorch>=1.5.0
- numpy
- pandas
- torch
- sklearn
- biotransformers
- biodatasets
- deepchain

## tasks
- transformers
- pca
- binary classification

## embeddings
- ESM1_t34_670M_UR100

## Author

Selim Jerad - Research Intern @ InstaDeep, Bachelor Student @ EPFL - s.jerad@instadeep.com

## Datasets / Resources

[HMD-ARG](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01002-3)  

[antibiotic-resistance](https://github.com/DeepChainBio/bio-datasets/blob/main/datasets/antibiotic-resistance/description.md)

[deepchain apps](https://github.com/DeepChainBio/deepchain-apps)

