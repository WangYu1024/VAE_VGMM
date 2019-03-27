

# # Cluster using VGMM

from sklearn import mixture
import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def cluster_vgmm(X_train):

 # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full').fit(X_train)
    X_prediction_vgmm = dpgmm.predict(X_train)
    print(dpgmm.means_.shape)
    print(dpgmm.means_)
    print(dpgmm.covariances_.shape)

 # dict for the data clustered into defferent groups
    dict={}
    for i in range(5):
        dict[str(i)]=np.where(X_prediction_vgmm==i)[0].tolist()

    return dict






