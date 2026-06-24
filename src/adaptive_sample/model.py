import numpy as np
import geopandas as gpd
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.affinity import translate
import pandas as pd

class MoistureModel:
    """
    Class to hold the GP for building a moisture map
    """
    def __init__(self):
        kernel = RBF(length_scale=40.0,length_scale_bounds=[10.0,800.0]) + 1.0 * WhiteKernel(noise_level_bounds=(1e-8, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            optimizer=None
        )
        self.X = np.zeros((0,2))
        self.y = np.zeros(0)
        self.obs_since_retrain = 0
    
    def add_observation(self,X_in,y_in,virtual=False):
        X_in = np.asarray(X_in, dtype=float).reshape(1, 2)
        y_in = float(np.asarray(y_in).item())
        if not virtual:
            self.obs_since_retrain += 1

        self.X = np.vstack([self.X, X_in])
        self.y = np.concatenate([self.y, [y_in]])

        # Fit the data, retraining hyperparameters if need be
        if (not virtual) and self.obs_since_retrain % 5 == 0:
            self.retrain_hyperparameters()
            self.obs_since_retrain = 0
        else:
            self.gp.fit(self.X, self.y)
    
    def retrain_hyperparameters(self):
        # We need an optimizer that will optimizer the kernel hyperparameters
        kernel = self.gp.kernel_ if hasattr(self.gp, "kernel_") else self.gp.kernel
        gp_opt = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=10,
            optimizer='fmin_l_bfgs_b'
        )
        gp_opt.fit(self.X,self.y)

        # Use the fit kernel
        self.gp = GaussianProcessRegressor(
            kernel=gp_opt.kernel_,
            normalize_y=True,
            optimizer=None
        )
        self.gp.fit(self.X, self.y)
