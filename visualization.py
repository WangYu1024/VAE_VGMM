# tsne plot for data after clustring


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def tsne_plot(X_train,y_train,dict):
    fashion_pca_tsne = {}
    for i in range (5):
        X = X_train[dict[str(i)]]
        y = y_train[dict[str(i)]]
        pca_50 = PCA(n_components=10)
        pca_result_50 = pca_50.fit_transform(X)
        fashion_pca_tsne[str(i)] = TSNE().fit_transform(pca_result_50)

    plt.scatter(fashion_pca_tsne['0'][:,0], fashion_pca_tsne['0'][:,1], color='navy',alpha =0.1)
    plt.scatter(fashion_pca_tsne['1'][:,0], fashion_pca_tsne['1'][:,1], color='c',alpha =0.1)
    plt.scatter(fashion_pca_tsne['2'][:,0], fashion_pca_tsne['2'][:,1], color='cornflowerblue',alpha =0.1)
    plt.scatter(fashion_pca_tsne['3'][:,0], fashion_pca_tsne['3'][:,1], color='gold',alpha =0.1)
    plt.scatter(fashion_pca_tsne['4'][:,0], fashion_pca_tsne['4'][:,1], color='darkorange',alpha =0.1)
