"""
Author: Alicia Curth
Module contains base learners for treatment effect estimation (as described in Curth, Alaa and
van der Schaar (2020), namely
- Plug-in learners (also known as T-learners in the CATE setting)
- IF-learners (also known as DR-learners in the CATE setting)
- HT-learners (learners using pseudo-outcomes based on the Horvitz-Thompson transformation)
- Oracle estimators
"""
import abc
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.base import clone, check_estimator_has_method, get_name_needed_prediction_method
from .transformations import CATE_NAME, _get_po_plugin_function, _get_te_eif, _get_ht_transformation


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

    @staticmethod
    def _check_inputs(w, p):
        if p is not None:
            if np.sum(p > 1) > 0 or np.sum(p < 0) > 0:
                raise ValueError('p should be in [0,1]')

        if not ((w == 0) | (w == 1)).all():
            raise ValueError('W should be binary')


class PlugInTELearner(BaseTEModel):
    """
    Class for Plug-in estimation of treatment effects.  For the CATE setting, this estimator is also
    known as the T-learner.


    Parameters
    ----------
    base_estimator: estimator
        Estimator to be used for potential outcome regressions
    setting: str or callable, default 'CATE'
        The treatment effect setting to be considered. Currently built-in support for 'CATE' or
        'RR'. Can also pass a callable that is a plug-in function of the two potenital outcome
        regressions.
    binary_y: bool, default False
        Whether the outcome data is binary
    """
    def __init__(self, base_estimator, setting=CATE_NAME, binary_y: bool = False):
        self.base_estimator = base_estimator
        self.setting = setting
        self.binary_y = binary_y

    def _prepare_self(self):
        needed_pred_method = get_name_needed_prediction_method(self.binary_y)
        check_estimator_has_method(self.base_estimator, needed_pred_method,
                                   'base_estimator', return_clone=False)
        # to make sure that we are starting with clean objects
        self._plug_in_0 = clone(self.base_estimator)
        self._plug_in_1 = clone(self.base_estimator)

        # set potential outcome function
        self._po_plugin_function = _get_po_plugin_function(self.setting, self.binary_y)

    def fit(self, X, y, w, p=None):
        """
        Fit plug-in models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The features to fit to
        y : array-like of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: array-like of shape (n_samples,)
            The treatment indicator
        p: array-like of shape (n_samples,)
            The treatment propensity
        """
        self._prepare_self()
        self._check_inputs(w, p)
        self._plug_in_0.fit(X[w == 0], y[w == 0])
        self._plug_in_1.fit(X[w == 1], y[w == 1])

    def predict(self, X, return_po: bool = False):
        """
        Predict treatment effects and potential outcomes

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        y_0: array-like of shape (n_samples,)
            Predicted Y(0)
        y_1: array-like of shape (n_samples,)
            Predicted Y(1)
        """
        if self.binary_y:
            y_0 = self._plug_in_0.predict_proba(X)
            y_1 = self._plug_in_1.predict_proba(X)
        else:
            y_0 = self._plug_in_0.predict(X)
            y_1 = self._plug_in_1.predict(X)

        te_est = self._po_plugin_function(mu_0=y_0, mu_1=y_1)
        if return_po:
            return te_est, y_0, y_1
        else:
            return te_est


class HTLearnerTE(BaseTEModel):
    """
    Class implementing a learner based on the Horvitz-Thompson transformed outcome

    Parameters
    ----------
    te_estimator: estimator
        estimator for second stage
    setting: str or callable, default 'CATE'
        The treatment effect setting to be considered. Currently built-in support for 'CATE' or
        'PO1' and 'PO0'. Can also pass a callable that is another transformation.
    fit_propensity_model: bool, default False
        Whether to fit a propensity model. If not, propensity scores have to be passed.
    propensity_estimator: estimator, default None
        estimator for propensity scores. Needed only if fit_propensity_model is True
    n_folds: int, default 10
        Number of cross-fitting folds
    random_state: int, default 42
        random state to use for cross-fitting splits
    """
    def __init__(self, te_estimator, propensity_estimator=None,
                 setting=CATE_NAME, fit_propensity_model: bool = False,
                 n_folds: int = 10, random_state: int = 42):
        # set estimators
        self.te_estimator = te_estimator
        self.fit_propensity_model = fit_propensity_model
        if fit_propensity_model and propensity_estimator is None:
            raise ValueError('Need to pass propensity estimator when it should be fitted.')
        self.propensity_estimator = propensity_estimator

        # set other arguments
        self.setting = setting
        self.n_folds = n_folds
        self.random_state = random_state

    def _prepare_self(self):
        # check that all estimators have the attributes they should have and clone them to be safe
        if self.propensity_estimator is not None:
            self.propensity_estimator = check_estimator_has_method(self.propensity_estimator,
                                                                   'predict_proba',
                                                                   'propensity_estimator')
        self._ht_transformation = _get_ht_transformation(self.setting)

    def _propensity_step(self, X, y, w, fit_mask, pred_mask):
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        temp_propensity_estimator = clone(self.propensity_estimator)
        temp_propensity_estimator.fit(X_fit, W_fit)
        p_pred = temp_propensity_estimator.predict_proba(X[pred_mask, :])
        return p_pred

    def _pseudo_outcome_step(self, X, y, w, p):
        # create transformed outcome based on efficient influence function
        transformed_outcome = self._ht_transformation(y, w, p)
        self.te_estimator.fit(X, transformed_outcome)

    def fit(self, X, y, w, p=None):
        """
        Fit two stages of pseudo-outcome regression to get treatment effect estimators

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The features to fit to
        y : array-like of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: array-like of shape (n_samples,)
            The treatment indicator
        p: array-like of shape (n_samples,)
            The treatment propensity
        """
        self._prepare_self()
        self._check_inputs(w, p)
        self._fit(X, y, w, p)

    def _fit(self, X, y, w, p=None):
        n = len(y)

        if p is None and not self.fit_propensity_model:
            # assume equal probabilities
            p = 0.5 * np.ones(n)

        # STEP 1: fit propensity estimator via cross-fitting
        if self.fit_propensity_model:
            if self.n_folds == 1:
                # no cross-fitting (not recommended)
                warnings.warn("You chose to not use cross-fitting. This is not recommended.")
                pred_mask = np.ones(n, dtype=bool)
                # fit plug-in te_estimator
                p_pred = self._propensity_step(X, y, w, pred_mask, pred_mask)

            else:
                p_pred = np.zeros(n)

                # create folds stratified by treatment assignment to ensure balance
                splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                           random_state=self.random_state)

                for train_index, test_index in splitter.split(X, w):
                    # create masks
                    pred_mask = np.zeros(n, dtype=bool)
                    pred_mask[test_index] = 1

                    # fit plug-in te_estimator
                    p_pred[pred_mask] = self._propensity_step(X, y, w, ~pred_mask, pred_mask)

        if self.fit_propensity_model:
            # use estimated propensity scores
            p = p_pred

        # STEP 2: bias correction
        self._pseudo_outcome_step(X, y, w, p)

    def predict(self, X, return_po=False):
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise ValueError("HT-learner does not support parameter return_po")

        return self.te_estimator.predict(X)


class IFLearnerTE(BaseTEModel):
    """
    Class implementing the IF-learner of Curth, Alaa and van der Schaar (2020) for the special
    case of treatment effect estimation. For the CATE setting, this estimator is also known as
    the DR-learner (Doubly robust learner).

    Parameters
    ----------
    te_estimator: estimator
        estimator for second stage
    base_estimator: estimator, default None
        estimator for first stage. Can be None, then te_estimator is used
    setting: str or callable, default 'CATE'
        The treatment effect setting to be considered. Currently built-in support for 'CATE' or
        'RR'. Can also pass a callable that is a plug-in function of the two potential outcome
        regressions.
    binary_y: bool, default False
        Whether the outcome data is binary
    fit_base_model: bool, default False
        Whether to fit a plug-in model on the full-sample to get potential outcome regressions
        in addition to treatment effect model
    fit_propensity_model: bool, default False
        Whether to fit a propensity model
    propensity_estimator: estimator, default None
        estimator for propensity scores. Needed only if fit_propensity_model is True
    n_folds: int, default 10
        Number of cross-fitting folds
    random_state: int, default 42
        random state to use for cross-fitting splits
    """
    def __init__(self, te_estimator, base_estimator=None, propensity_estimator=None,
                 setting=CATE_NAME, binary_y: bool = False,
                 fit_base_model: bool = False, fit_propensity_model: bool = False,
                 n_folds: int = 10, random_state: int = 42):
        # set estimators
        if te_estimator is None and base_estimator is None:
            raise ValueError('Need a te_estimator or a base_estimator')

        if te_estimator is None:
            self.te_estimator = base_estimator
        else:
            self.te_estimator = te_estimator
        if base_estimator is None:
            self.base_estimator = te_estimator
        else:
            self.base_estimator = base_estimator

        self.fit_propensity_model = fit_propensity_model
        if fit_propensity_model and propensity_estimator is None:
            raise ValueError('Need to pass propensity estimator when it should be fitted.')
        self.propensity_estimator = propensity_estimator

        # set other arguments
        self.setting = setting
        self.fit_base_model = fit_base_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.binary_y = binary_y

    def _prepare_self(self):
        # check that all estimators have the attributes they should have and clone them to be safe
        if self.propensity_estimator is not None:
            self.propensity_estimator = check_estimator_has_method(self.propensity_estimator,
                                                                   'predict_proba',
                                                                   'propensity_estimator')

        needed_pred_method = get_name_needed_prediction_method(self.binary_y)
        self.base_estimator = check_estimator_has_method(self.base_estimator,
                                                         needed_pred_method,
                                                         'base_estimator')

        self.te_estimator = check_estimator_has_method(self.te_estimator,
                                                       'predict',
                                                       'te_estimator')

        if self.fit_base_model:
            self._plug_in_0 = clone(self.base_estimator)
            self._plug_in_1 = clone(self.base_estimator)

        self._eif = _get_te_eif(self.setting, self.binary_y)

    def _plug_in_step(self, X, y, w, fit_mask, pred_mask):
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        # untreated model
        temp_model_0 = clone(self.base_estimator)
        temp_model_0.fit(X_fit[W_fit == 0], Y_fit[W_fit == 0])

        # treated model
        temp_model_1 = clone(self.base_estimator)
        temp_model_1.fit(X_fit[W_fit == 1], Y_fit[W_fit == 1])

        if self.binary_y:
            mu_0_pred = temp_model_0.predict_proba(X[pred_mask, :])
            mu_1_pred = temp_model_1.predict_proba(X[pred_mask, :])
        else:
            mu_0_pred = temp_model_0.predict(X[pred_mask, :])
            mu_1_pred = temp_model_1.predict(X[pred_mask, :])

        if not self.fit_propensity_model:
            return mu_0_pred, mu_1_pred, np.nan
        else:
            # also get estimated propensity scores
            temp_propensity_estimator = clone(self.propensity_estimator)
            temp_propensity_estimator.fit(X_fit, W_fit)
            p_pred = temp_propensity_estimator.predict_proba(X[pred_mask, :])
            return mu_0_pred, mu_1_pred, p_pred

    def _bias_correction_step(self, X, y, w, p, mu_0, mu_1):
        # create transformed outcome based on efficient influence function
        transformed_outcome = self._eif(y, w, p, mu_0, mu_1)
        self.te_estimator.fit(X, transformed_outcome)

    def fit(self, X, y, w, p=None):
        """
        Fit two stages of pseudo-outcome regression to get treatment effect estimators

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The features to fit to
        y : array-like of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: array-like of shape (n_samples,)
            The treatment indicator
        p: array-like of shape (n_samples,)
            The treatment propensity
        """
        self._prepare_self()
        self._check_inputs(w, p)
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
            mu_0_pred, mu_1_pred, p_pred = self._plug_in_step(X, y, w, pred_mask, pred_mask)

        else:
            mu_0_pred, mu_1_pred, p_pred = np.zeros(n), np.zeros(n), np.zeros(n)

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                       random_state=self.random_state)

            for train_index, test_index in splitter.split(X, w):
                # create masks
                pred_mask = np.zeros(n, dtype=bool)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                mu_0_pred[pred_mask], mu_1_pred[pred_mask], p_pred[pred_mask] = \
                    self._plug_in_step(X, y, w, ~pred_mask, pred_mask)

        if self.fit_propensity_model:
            # use estimated propensity scores
            p = p_pred

        # STEP 2: bias correction
        self._bias_correction_step(X, y, w, p, mu_0_pred, mu_1_pred)

        if self.fit_base_model:
            # also fit a single baseline model to return PO predictions later
            self._plug_in_0.fit(X[w == 0], y[w == 0])
            self._plug_in_1.fit(X[w == 1], y[w == 1])

    def predict(self, X, return_po=False):
        """
        Predict treatment effects and potential outcomes

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        y_0: array-like of shape (n_samples,)
            Predicted Y(0)
        y_1: array-like of shape (n_samples,)
            Predicted Y(1)
        """
        if return_po:
            # return both treatment effect estimates and potential outcome (first stage) estimates
            if not self.fit_base_model:
                raise ValueError("Cannot return potential outcomes when no base-model is fit")
            else:
                te_est = self.te_estimator.predict(X)
                if self.binary_y:
                    y_0 = self._plug_in_0.predict_proba(X)
                    y_1 = self._plug_in_1.predict_proba(X)
                else:
                    y_0 = self._plug_in_0.predict(X)
                    y_1 = self._plug_in_1.predict(X)
                return te_est, y_0, y_1
        else:
            # return only
            return self.te_estimator.predict(X)


class TEOracle(BaseTEModel):
    """
    Class which implements a treatment effect oracle with knowledge of all underlying functions.

    Parameters
    ----------
    te_model: callable
        function outputting treatment effects as a function of X
    base_model: callable
        function outputting baseline outcomes as a function of X
    """
    def __init__(self, te_model, base_model, setting=CATE_NAME, binary_y: bool = False):
        self.te_model = te_model
        self.base_model = base_model
        self.setting = setting
        self.binary_y = binary_y

    def fit(self, X, y, w, p=None):
        # set potential outcome function
        self._po_plugin_function = _get_po_plugin_function(self.setting, self.binary_y)

    def predict(self, X, return_po=False):
        """
        Give treatment effects and potential outcomes using Oracle knowledge

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te: array-like of shape (n_samples,)
            Oracle treatment effects
        mu_0: array-like of shape (n_samples,)
            Oracle E[Y(0)|X]
        mu_1: array-like of shape (n_samples,)
            Oracle E[Y(1)|X]
        """
        mu_0 = self.base_model(X)
        mu_1 = mu_0 + self.te_model(X)
        te = self._po_plugin_function(mu_0=mu_0, mu_1=mu_1)
        if return_po:
            return te, mu_0, mu_1
        else:
            return te


class IFTEOracle(BaseTEModel):
    """
    Class to estimate treatment effects using IF-learner second stage and using oracle first stage

    Parameters
    ----------
    te_estimator: estimator
        estimator for second stage
    te_model: callable
        function outputting treatment effects as a function of X
    base_model: callable
        function outputting baseline outcomes as a function of X
    """

    def __init__(self, te_estimator, te_model, base_model, setting=CATE_NAME):
        # override super
        self.te_estimator = te_estimator
        self.te_model = te_model
        self.base_model = base_model
        self.setting = setting

    def fit(self, X, y, w, p=None):
        """
        Fit second stage of IF-estimator given first stage oracle

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The features to fit to
        y : array-like of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: array-like of shape (n_samples,)
            The treatment indicator
        p: array-like of shape (n_samples,)
            The treatment propensity
        """
        self._check_inputs(w, p)

        # set EIF
        self._eif = _get_te_eif(self.setting)

        # create pseudo outcome
        mu_0 = self.base_model(X)
        mu_1 = mu_0 + self.te_model(X)
        pseudo_outcome = self._eif(y, w, p, mu_0, mu_1)

        # fit pseudo model
        self.te_estimator.fit(X, pseudo_outcome)

    def predict(self, X, return_po=False):
        """
        Predict treatment effects and potential outcomes using fitted te-model

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        mu_0: array-like of shape (n_samples,)
            Oracle E[Y(0)|X]
        mu_1: array-like of shape (n_samples,)
            Oracle E[Y(1)|X]
        """
        te = self.te_estimator.predict(X)
        mu_0 = self.base_model(X)
        mu_1 = mu_0 + self.te_model(X)
        if return_po:
            return te, mu_0, mu_1
        else:
            return te
