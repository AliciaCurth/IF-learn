""""
Author: Alicia Curth
Scoring function for when treatment effects are known (oracle knowledge). This is useful mainly for
simulation studies.
"""
import numpy as np
import time
import warnings
import numbers

from traceback import format_exc

from sklearn.utils.validation import _check_fit_params
from sklearn.utils import _safe_indexing
from sklearn.model_selection._validation import _score
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning

from .base_learners import BaseTEModel
from ..utils.base import clone


def fit_and_score_te_oracle(estimator, X, y, w, p, t, scorer, train, test,
                            parameters=None, fit_params=None, return_train_score=False,
                            return_parameters=False,
                            return_times=False, return_estimator=False,
                            error_score=np.nan,
                            return_test_score_only=False):
    """Fit estimator and compute scores for a given dataset split. Based on
    sklearn.model_selection._validation _fit_and_score.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    w: array-like of shape (n_samples,)
        the treatment indicator
    p: array-like of shape (n_samples,)
        the treatment propensity
    t: array-like of shape (n_samples,)
        the true treatment effect to evaluate against
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    return_test_score_only: bool, default=False
        Whether to only return a test score

    Returns
    -------
    train_scores : dict of scorer name -> float
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.
    test_scores : float or dict of scorer name -> float
        If return_test_score_only and scorer == str, then returns only test score. Otherwise,
        s on testing set (for all the scorers)
    n_test_samples : int
        Number of test samples.
    fit_time : float
        Time spent for fitting in seconds.
    score_time : float
        Time spent for scoring in seconds.
    parameters : dict or None
        The parameters that have been evaluated.
    estimator : estimator object
        The fitted estimator
    """
    if not isinstance(estimator, BaseTEModel):
        raise ValueError("This method works only for BaseTEModel")

    scorers, _ = _check_multimetric_scoring(estimator, scoring=scorer)

    # Adjust length of sample weights (if ant)
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    train_scores = {}
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train, w_train, p_train, t_train = _safe_split_te(X, y, w, p, t, train)
    X_test, y_test, w_test, p_test, t_test = _safe_split_te(X, y, w, p, t, test)

    try:
        estimator.fit(X_train, y_train, w_train, p_train, **fit_params)

    except Exception as e:
        if return_test_score_only:
            if error_score == 'raise':
                raise
            else:
                return np.nan
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, t_test, scorers)
        score_time = time.time() - start_time - fit_time

        if return_test_score_only:
            if type(scorer) == str:
                return test_scores['score']
            else:
                return test_scores

        if return_train_score:
            train_scores = _score(estimator, X_train, t_train, scorers)

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret


def _safe_split_te(X, y, w, p, t, indices):
    """ Split data for treatment effect estimation. Based on sklearn's safe split in
    sklearn.utils.metaestimators

    Parameters
    ----------
    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.
    y : array-like, sparse matrix or iterable
        Targets to be indexed.
    w: array-like of shape (n_samples,)
        the treatment indicator
    p: array-like of shape (n_samples, )
        the treatment propensity
    t: array-like of shape (n_samples,)
        the truth to evaluate against (can be None)
    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.
    y_subset : array-like, sparse matrix or list
        Indexed targets.
    """
    X_subset = _safe_indexing(X, indices)
    y_subset = _safe_indexing(y, indices)
    w_subset = _safe_indexing(w, indices)

    if t is not None:
        t_subset = _safe_indexing(t, indices)
    else:
        t_subset = None

    if p is not None:
        p_subset = _safe_indexing(p, indices)
    else:
        p_subset = None

    return X_subset, y_subset, w_subset, p_subset, t_subset
