"""
Author: Alicia Curth
Simulation utils for treatment effect estimation simulations
"""
import numpy as np

from .base import BASE_BASELINE_MODEL, BASE_COVARIATE_MODEL, BASE_ERROR_MODEL, constant_baseline
from ..treatment_effects.base import _get_po_plugin_function, CATE_NAME
BASE_TE_MODEL = constant_baseline


def make_te_data(n: int = 1000, d: int = 1, setting=CATE_NAME, covariate_model=None,
                 binary=False, te_model=None,
                 propensity_model=None,
                 baseline_model=None, noise=True, error_model=None,
                 selection_bias=None, seedy=42):
    """
    Simulate data for treatment effect estimation

    Parameters
    ----------
    n
    d
    setting
    covariate_model
    binary
    te_model
    propensity_model
    baseline_model
    noise
    error_model
    seedy
    selection_bias

    Returns
    -------

    """
    # set seed
    if seedy:
        np.random.seed(seedy)

    if covariate_model is not None:
        X = covariate_model(n, d)
    else:
        X = BASE_COVARIATE_MODEL(n, d)

    if selection_bias is not None:
        # simulate data according to selection bias model
        p = selection_bias(X)
        w = np.random.binomial(1, p=p)
    else:
        # simulate data according to propensity model
        if propensity_model is None:
            # equal treatment propensities
            w = np.random.choice([0, 1], n)
            p = 0.5 * np.ones(n)
        else:
            p = propensity_model(X)
            w = np.random.binomial(1, p=p)

    if te_model is None:
        t = BASE_TE_MODEL(X)
    else:
        t = te_model(X)

    # add error
    if binary:
        y = w*t
    else:
        if noise:
            if error_model is None:
                y = w * t + BASE_ERROR_MODEL(X)
            else:
                y = w * t + error_model(X)
        else:
            y = w * t

    # get baseline
    if baseline_model is not None:
        # add baseline effects
        bs = baseline_model(X)
    else:
        bs = BASE_BASELINE_MODEL(X)

    # generate outcome
    if binary:
        y = np.random.binomial(1, p=y+bs)
    else:
        y = y + bs

    if selection_bias is not None:
        # switch out propensity scores
        if propensity_model is None:
            p = 0.5 * np.ones(n)
        else:
            p = propensity_model(X)

    # compute the treatment effect transformation we want
    po_function = _get_po_plugin_function(setting, binary)
    ite = po_function(bs, bs+t)

    return X, y, w, ite, p, bs


def simple_propensity_model(X, dim: int = 0):
    """
    Propensity score used in Kennedy (2020)

    Parameters
    ----------
    X
    dim

    Returns
    -------

    """
    return 0.1 * np.ones(X.shape[0]) + 0.8 * (X[:, dim] > 0)

