"""
Author: Alicia Curth
Simulation utils for treatment effect estimation simulations
"""
import numpy as np
from scipy.stats import beta

from .base import BASE_BASELINE_MODEL, BASE_COVARIATE_MODEL, BASE_ERROR_MODEL, constant_baseline,\
    _get_values_only, _check_is_callable, baseline_wa
from ..treatment_effects.base import _get_po_plugin_function, CATE_NAME
BASE_TE_MODEL = constant_baseline


def make_te_data(n: int = 1000, d: int = 1, setting=CATE_NAME, covariate_model=None,
                 binary_y=False, te_model=None,
                 propensity_model=None,
                 baseline_model=None, noise: bool = True, error_model=None,
                 selection_bias=None, seedy: int = 42):
    """
    Simulate data for treatment effect estimation

    Parameters
    ----------
    n: int, default 1000
        number of observations to generate
    d: int, default 1
        number of dimensions of X to generate
    setting: str, default 'CATE'
        which treatment effect setting to consider. Currently supports 'RR' and 'CATE'
    covariate_model: callable
        which covariate model to use. Defaults to uniform [-1,1]
    binary_y: boolean, default False
        Whether outcome data is binary
    te_model: callable
        treatment effect model to use. Defaults to tau(x) = 0
    propensity_model: callable
        propensity model to use. Defaults to pi(x)=0.5
    baseline_model: callable
        baseline function to use. Defaults to the baseline used in Kennedy(2020)
    noise: bool, True
        whether to add noise to the outcomes
    error_model: callable
        error model to use. defaults to the model used in Kennedy(2020)
    seedy: int, default 42
        seed to use
    selection_bias: callable, defaults to None
        If selection bias is passed, then it will be treated as the propensity score,
        yet the simulator will return the incorrect propensity scores as provided by
        propensity_model(X)

    Returns
    -------
    X, y, w, ite, p, bs: covariates, outcomes, treatment_indicators, the true effect,
                        propensity scores and the true baseline
    """
    # set seed
    if seedy:
        np.random.seed(seedy)

    if covariate_model is not None:
        X = covariate_model(n=n, d=d)
    else:
        X = BASE_COVARIATE_MODEL(n, d)

    if selection_bias is not None:
        # simulate data according to selection bias model
        p = selection_bias(X=X)
        w = np.random.binomial(1, p=p)
    else:
        # simulate data according to propensity model
        if propensity_model is None:
            # equal treatment propensities
            w = np.random.choice([0, 1], n)
            p = 0.5 * np.ones(n)
        else:
            p = propensity_model(X=X)
            w = np.random.binomial(1, p=p)

    if te_model is None:
        t = BASE_TE_MODEL(X)
    else:
        t = te_model(X=X)

    # add error
    if binary_y:
        y = w*t
    else:
        if noise:
            if error_model is None:
                y = w * t + BASE_ERROR_MODEL(X)
            else:
                y = w * t + error_model(X=X)
        else:
            y = w * t

    # get baseline
    if baseline_model is not None:
        # add baseline effects
        bs = baseline_model(X=X)
    else:
        bs = BASE_BASELINE_MODEL(X)

    # generate outcome
    if binary_y:
        y = np.random.binomial(1, p=y+bs)
    else:
        y = y + bs

    if selection_bias is not None:
        # switch out propensity scores
        if propensity_model is None:
            p = 0.5 * np.ones(n)
        else:
            p = propensity_model(X=X)

    # compute the treatment effect transformation we want
    po_function = _get_po_plugin_function(setting, binary_y)
    ite = po_function(bs, bs+t)

    return X, y, w, ite, p, bs


# Models from other papers -------------------------------------------------------
# Kennedy (2020) -----------------------------------------------------------------
def simple_propensity_model(X, dim: int = 0):
    """
    Propensity score used in Kennedy (2020)

    Parameters
    ----------
    X: array-like
        input data to use
    dim: int, default 0
        Dimension of X to use

    Returns
    -------

    """
    X = _get_values_only(X)
    return 0.1 * np.ones(X.shape[0]) + 0.8 * (X[:, dim] > 0)


# Wager & Athey (2018), Athey et al (2019) -----------------------------------------
def _nonlinear_effect_wa1(x):
    return 1 + 1/(1 + np.exp((-20*(x-1/3))))


def nonlinear_treatment_effect_wa1(X, dim_0=0, dim_1=1):
    """
    First nonlinear treatment effect from Wager & Athey (2018)

    Parameters
    ----------
    X: array-like
        input data to use
    dim_1, dim_0: int, default 0 and 1
        Dimensions of X to use
    """
    X = _get_values_only(X)
    return _nonlinear_effect_wa1(X[:, dim_0]) * _nonlinear_effect_wa1(X[:, dim_1])


def _nonlinear_effect_wa2(x):
    return 1/(1 + np.exp((-12*(x-1/2))))


def nonlinear_treatment_effect_wa2(X, dim_0=0, dim_1=1):
    """
    Second nonlinear treatment effect from Wager & Athey (2018)

    Parameters
    ----------
    X: array-like
        input data to use
    dim_1, dim_0: int, default 0 and 1
        Dimensions of X to use
    """
    X = _get_values_only(X)
    return _nonlinear_effect_wa2(X[:, dim_0]) * _nonlinear_effect_wa2(X[:, dim_1])


def propensity_wa(X, dim: int = 2):
    """
    Propensity score from Wager & Athey (2018)

    Parameters
    ----------
    X: array-like
        input data to use
    dim: int, default 2
        Dimension of X to use
    """
    X = _get_values_only(X)
    return 0.25 * (beta.pdf(X[:, dim], 2, 4) + 1)


# New multiplicative variations on treatment effects -----------------------------
def te_multiple_baseline(X, baseline=None, multiplier: float = 3):
    """
    Create treatment effect that is a multiple of the baseline function

    Parameters
    ----------
    X: array-like
        input data to use
    baseline: callable
        Baseline function. Defaults to the one used in Wager & Athey (2018)
    multiplier: float, default 3
        multiplier to use
    """
    if baseline is None:
        baseline = baseline_wa
    else:
        # check is callable
        _check_is_callable(baseline, 'baseline function')
    X = _get_values_only(X)
    return multiplier * baseline(X)


def te_interaction_baseline(X, other_function=None, baseline=None):
    """
    Create a treatment effect that has the form tau(x) = nonlin(x)*baseline(x)
    Parameters
    ----------
    X: array-like
        input data to use
    other_function: callable
        Effect function to interact with baseline
    baseline: callable
        Baseline function. Defaults to the one used in Wager & Athey (2018)
    """
    if baseline is None:
        baseline = baseline_wa
    else:
        # check is callable
        _check_is_callable(baseline, 'baseline function')

    if other_function is None:
        other_function = nonlinear_treatment_effect_wa1
    else:
        # check is callable
        _check_is_callable(other_function, 'other function')
    X = _get_values_only(X)
    return other_function(X) * baseline(X)
