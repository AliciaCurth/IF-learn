"""
Author: Alicia Curth
Module implements helper classes for simulations. In particular, this module contains:
(1) AdaptiveLogisticGAM, a LogisticGAM with internal hyperparameter optimixation
(2) RSmoothingSpline, Base Rs smooth.splinein in a sklearn-style python wrapper
(3) RRegressionForest, RegressionForest from the R package grf in a
 sklearn-style python wrapper
 (4) RCausalForest, Causal forest from the R package grf in a
  sklearn-style python wrapper
"""
import pandas as pd
import numpy as np

from pygam import LogisticGAM
from sklearn.base import BaseEstimator, RegressorMixin

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri

from iflearn.treatment_effects.base_learners import BaseTEModel

rpy2.robjects.numpy2ri.activate()
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
        y_spline = np.array(robjects.r['predict'](self._spline,
                            robjects.FloatVector(X)).rx2('y'))
        return y_spline


# check if package is installed
def _importr_tryhard(packname):
    from rpy2.rinterface import RRuntimeError
    utils = importr('utils')
    try:
        rpack = importr(packname)
    except RRuntimeError:
        utils.install_packages(packname)
        rpack = importr(packname)
    return rpack


class RRegressionForest(BaseEstimator, RegressorMixin):
    """
    Wrapper for Athey, Tibshirani & Wager (2019)'s random forest as implemented in the GRF
    package.
    """
    def __init__(self, n_estimators: int = 2000, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit regression forest
        """
        # bring data in right shape
        if isinstance(X, pd.DataFrame):
            X = X.values
        n, d = X.shape
        r_y = robjects.FloatVector(y)
        r_x = robjects.r.matrix(X, n, d)

        # get forest
        self._grf = _importr_tryhard("grf")
        self._estimator = self._grf.regression_forest(X=r_x, Y=r_y,
                                                      num_trees=self.n_estimators,
                                                      seed=self.random_state)

    def predict(self, X, return_se: bool = False):
        """
        Make prediction, with or without standard errors associated
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        n, d = X.shape
        r_x = robjects.r.matrix(X, n, d)

        if return_se:
            # predict with var
            r_pred = self._grf.predict_regression_forest(self._estimator, newdata=r_x,
                                                         estimate_variance=True)
            r_pred = np.transpose(pandas2ri.ri2py_dataframe(r_pred).values)
            return r_pred[:, 0], r_pred[:, 1]
        else:
            r_pred = self._grf.predict_regression_forest(self._estimator, newdata=r_x)
            r_pred = pandas2ri.ri2py_dataframe(r_pred).values
            return np.transpose(r_pred[0, :])


class RCausalForest(BaseTEModel):
    """
    Wrapper for Athey, Tibshirani & Wager (2019)'s random forest as implemented in the GRF
    package.
    """
    def __init__(self, n_estimators: int = 2000, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y, w, p=None):
        """
        Fit causal forest
        """
        # bring data in right shape
        if isinstance(X, pd.DataFrame):
            X = X.values

        n, d = X.shape
        r_y = robjects.FloatVector(y)
        r_x = robjects.r.matrix(X, n, d)
        r_w = robjects.IntVector(w)

        # get forest
        self._grf = _importr_tryhard("grf")

        if p is not None:
            # give propensity estimator
            r_p = robjects.FloatVector(p)
            params = {'X': r_x, 'Y': r_y, 'W': r_w, 'W.hat': r_p,
                      'num_trees': self.n_estimators, 'seed': self.random_state}
            self._estimator = self._grf.causal_forest(**params)

        else:
            self._estimator = self._grf.causal_forest(X=r_x, Y=r_y, W=r_w,
                                                      num_trees=self.n_estimators,
                                                      seed=self.random_state)

    def predict(self, X, return_po: bool = False, return_se: bool = False):
        """
        Make prediction, with or without standard errors associated

        Parameters
        ----------
        X: array-like
            Test data
        return_se: bool
            Whether to return standard errors

        Returns
        -------

        """
        if return_po:
            raise NotImplementedError('Causal forest does not return potential outcomes')

        if isinstance(X, pd.DataFrame):
            X = X.values
        n, d = X.shape
        r_x = robjects.r.matrix(X, n, d)

        if return_se:
            # predict with var
            r_pred = self._grf.predict_causal_forest(self._estimator, newdata=r_x,
                                                     estimate_variance=True)
            r_pred = np.transpose(pandas2ri.ri2py_dataframe(r_pred).values)
            return r_pred[:, 0], r_pred[:, 1]
        else:
            r_pred = self._grf.predict_causal_forest(self._estimator, newdata=r_x)
            r_pred = pandas2ri.ri2py_dataframe(r_pred).values
            return np.transpose(r_pred[0, :])
