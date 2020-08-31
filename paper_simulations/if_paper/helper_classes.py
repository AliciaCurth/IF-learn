"""
Author: Alicia Curth
Module implements helper classes for simulations
"""
import pandas as pd
import numpy as np

from pygam import LogisticGAM
from sklearn.base import BaseEstimator, RegressorMixin

import rpy2.robjects as robjects

GAM_GRID_BASE = {'lam': [0.001, 0.1, 0.6, 1, 10]}


# Adaptive splines and GAMs -------------------------------------------------------------
class AdaptiveLogisticGAM(BaseEstimator, RegressorMixin):
    def __init__(self, param_grid=None, gam_params=None):
        # create GAM
        if gam_params is None:
            gam_params = {}
        self.model = LogisticGAM(**gam_params)

        # set grid search parameters
        if param_grid is None:
            param_grid = GAM_GRID_BASE
        self.param_grid = param_grid

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # fit using grid-search
        self.model.gridsearch(X, y, progress=False, **self.param_grid)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)


# --------------------------------- R's smooth spline -------------------------
def r_nknots_spline(n):
    # taken from R
    if n < 50:
        return n
    else:
        a1 = np.log2(50)
        a2 = np.log2(100)
        a3 = np.log2(140)
        a4 = np.log2(200)
        if n < 200:
            return int(2**(a1 + (a2-a1)*(n-50)/150))
        elif n < 800:
            return int(2**(a2 + (a3-a2)*(n-200)/600))
        elif n < 3200:
            return int(2**(a3 + (a4-a3)**(n-800)/2400))
        else:
            return int(200 + (n-3200)**0.2)


class RSmoothingSpline(BaseEstimator, RegressorMixin):
    """
    Wrapper for base R's smooth.spline
    """
    def __init__(self, nknots=None):
        self.nknots = nknots

    def fit(self, X, y):
        r_y = robjects.FloatVector(y)
        r_x = robjects.FloatVector(X)

        r_smooth_spline = robjects.r['smooth.spline']  # extract R function

        if self.nknots is None:
            self.nknots = r_nknots_spline(len(y))

        self._spline = r_smooth_spline(x=r_x, y=r_y, nknots=self.nknots)

    def predict(self, X):
        y_spline = np.array(robjects.r['predict'](self._spline, robjects.FloatVector(X)).rx2('y'))
        return y_spline


# Selection bias class ---------------------------------------------
class TwoSideBias:
    """
    Class implements selection bias as
    """
    def __init__(self, b=1, dim: int = 0):
        self.b = b
        self.dim = dim

    def __call__(self, X, *args, **kwargs):
        return 0.5 + 0.5 * self.b/2 * np.sign(X[:, self.dim]) * X[:, self.dim] - 0.5 * self.b/2 * \
               np.sign(-X[:, self.dim]) * X[:, self.dim]
