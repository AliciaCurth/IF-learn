"""
Author: Alicia Curth
Module contains base learners for treatment effect estimation (as described in Curth, Alaa and
van der Schaar (2020), namely
- Plug-in learners (also known as T-learners in the CATE setting)
- IF-learners (also known as DR-learners in the CATE setting)
- Oracle estimators
"""
import abc
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.base import clone
from .base import CATE_NAME, _get_po_function, _get_te_eif


class BaseTEModel(BaseEstimator, RegressorMixin, abc.ABC):
    """
    Base class for treatment effect models
    """
    def __init__(self):
        pass

    def score(self, X, y, sample_weight=None):
        pass

    @abc.abstractmethod
    def fit(self, X, y, w, p=None):
        pass

    @abc.abstractmethod
    def predict(self, X, return_po=False):
        pass


class PlugInTELearner(BaseTEModel):
    # ML-based TE-model based on a T-learner
    def __init__(self, base_estimator, setting=CATE_NAME, binary=False):
        self.base_estimator = base_estimator
        self.setting = setting
        self.binary = binary

    def _prepare_self(self):
        # to make sure that we are starting with clean objects
        self._plug_in_0 = clone(self.base_estimator)
        self._plug_in_1 = clone(self.base_estimator)

        # set potential outcome function
        self._po_function = _get_po_function(self.setting, self.binary)

    def fit(self, X, y, w, p=None):
        self._prepare_self()
        self._plug_in_0.fit(X[w == 0], y[w == 0])
        self._plug_in_1.fit(X[w == 1], y[w == 1])

    def predict(self, X, return_po=False):
        if self.binary:
            Y_est_0 = self._plug_in_0.predict_proba(X)
            Y_est_1 = self._plug_in_1.predict_proba(X)
        else:
            Y_est_0 = self._plug_in_0.predict(X)
            Y_est_1 = self._plug_in_1.predict(X)

        TE_est = self._po_function(mu_0=Y_est_0, mu_1=Y_est_1)
        if return_po:
            return TE_est, Y_est_0, Y_est_1
        else:
            return TE_est


class IFLearnerTE(BaseTEModel):
    def __init__(self, te_estimator, base_estimator=None, propensity_estimator=None,
                 double_sample_split=False, setting=CATE_NAME, binary=False,
                 fit_base_model=False, fit_propensity_model=False,
                 n_folds=10, random_state=42):

        # set estimators
        if te_estimator is None and base_estimator is None:
            raise ValueError('Need an te_estimator')

        if te_estimator is None:
            self.te_estimator = base_estimator
        else:
            self.te_estimator = te_estimator
        if base_estimator is None:
            self.base_estimator = te_estimator
        else:
            self.base_estimator = base_estimator

        # set other arguments
        self.setting = setting
        self.fit_base_model = fit_base_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.binary = binary

        # TODO add selection bias
        self.fit_propensity_model = fit_propensity_model
        self.double_sample_split = double_sample_split
        self.propensity_estimator = propensity_estimator

    def _prepare_self(self):
        # clone all estimators to be safe
        self.te_estimator = clone(self.te_estimator)
        self.base_estimator = clone(self.base_estimator)

        if self.fit_base_model:
            self._plug_in_0 = clone(self.base_estimator)
            self._plug_in_1 = clone(self.base_estimator)

        self._eif = _get_te_eif(self.setting, self.binary)

    def _plug_in_step(self, X, y, w, fit_mask, pred_mask):
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        # untreated model
        temp_model_0 = clone(self.base_estimator)
        temp_model_0.fit(X_fit[W_fit == 0], Y_fit[W_fit == 0])

        # treated model
        temp_model_1 = clone(self.base_estimator)
        temp_model_1.fit(X_fit[W_fit == 1], Y_fit[W_fit == 1])

        if self.binary:
            mu_0_pred = temp_model_0.predict_proba(X[pred_mask, :])
            mu_1_pred = temp_model_1.predict_proba(X[pred_mask, :])
        else:
            mu_0_pred = temp_model_0.predict(X[pred_mask, :])
            mu_1_pred = temp_model_1.predict(X[pred_mask, :])

        if not self.fit_propensity_model:
            return mu_0_pred, mu_1_pred
        else:
            raise NotImplementedError('Estimation of propensity scores not yet implemented')

    def _bias_correction_step(self, X, y, w, p, mu_0, mu_1):
        # create transformed outcome based on efficient influence function
        transformed_outcome = self._eif(y, w, p, mu_0, mu_1)
        self.te_estimator.fit(X, transformed_outcome)

    def fit(self, X, y, w, p=None):
        self._prepare_self()
        self._fit(X, y, w, p)

    def _fit(self, X, y, w, p=None):
        n = len(y)

        if p is None:
            # assume equal probabilities
            p = 0.5 * np.ones(n)

        # STEP 1: fit plug-in te_estimator via cross-fitting
        if self.n_folds == 1:
            # no cross-fitting (not recommended)
            warnings.warn("You chose to not use cross-fitting. This is not recommended.")
            pred_mask = np.ones(n, dtype=bool)
            # fit plug-in te_estimator
            mu_0_pred, mu_1_pred = self._plug_in_step(X, y, w, pred_mask, pred_mask)
        else:
            mu_0_pred, mu_1_pred = np.zeros(n), np.zeros(n)

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                       random_state=self.random_state)

            for train_index, test_index in splitter.split(X, w):
                # create masks
                pred_mask = np.zeros(n, dtype=bool)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                mu_0_pred[pred_mask], mu_1_pred[pred_mask] = self._plug_in_step(X, y, w,
                                                                                ~pred_mask,
                                                                                pred_mask)

        # STEP 2: bias correction
        self._bias_correction_step(X, y, w, p, mu_0_pred, mu_1_pred)

        if self.fit_base_model:
            # also fit a single baseline model to return PO predictions later
            self._plug_in_0.fit(X[w == 0], y[w == 0])
            self._plug_in_1.fit(X[w == 1], y[w == 1])

    def predict(self, X, return_po=False):
        if return_po:
            # return both treatment effect estimates and potential outcome (first stage) estimates
            if not self.fit_base_model:
                raise ValueError("Cannot return potential outcomes when no base-model is fit")
            else:
                te = self.te_estimator.predict(X)
                if self.binary:
                    y_0 = self._plug_in_0.predict_proba(X)
                    y_1 = self._plug_in_1.predict_proba(X)
                else:
                    y_0 = self._plug_in_0.predict(X)
                    y_1 = self._plug_in_1.predict(X)
                return te, y_0, y_1
        else:
            # return only
            return self.te_estimator.predict(X)


class TEOracle(BaseTEModel):
    """
    Class to be able to pass the truth into optimizers
    """

    def __init__(self, te_model, base_model):
        # override super
        self.te_model = te_model
        self.base_model = base_model

    def fit(self, X, y, w, p=None):
        # placeholder to fit syntax
        pass

    def predict(self, X, return_po=False):
        te = self.te_model(X)

        if return_po:
            mu_0 = self.base_model(X)
            return te, mu_0, mu_0 + te
        else:
            return te


class IFTEOracle(BaseTEModel):
    """
    Class to estimate IF using oracle first stage
    """

    def __init__(self, te_estimator, te_model, base_model, setting=CATE_NAME):
        # override super
        self.te_estimator = te_estimator
        self.te_model = te_model
        self.base_model = base_model
        self.setting = setting

    def fit(self, X, y, w, p=None):
        # set EIF
        self._eif = _get_te_eif(self.setting)

        # create pseudo outcome
        mu_0 = self.base_model(X)
        mu_1 = mu_0 + self.te_model(X)
        pseudo_outcome = self._eif(y, w, p, mu_0, mu_1)

        # fit pseudo model
        self.te_estimator.fit(X, pseudo_outcome)

    def predict(self, X, return_po=False):
        te = self.te_estimator.predict(X)
        mu_0 = self.base_model(X)
        mu_1 = mu_0 + self.te_model(X)
        if return_po:
            return te, mu_0, mu_1
        else:
            return te
