"""
Author: Alicia Curth
Simulation utils for generic simulation studies
"""
import numpy as np
import pandas as pd


# Baseline specifications ---------------------------------------------------------------
def constant_baseline(X, c: float = 0):
    """
    Create a constant baseline function

    Parameters
    ----------
    X: array-like
    c: float, default 0

    Returns
    -------
    A value for each row of X
    """
    return c * np.zeros(X.shape[0])


def gyorfi_baseline(X, dim: int = 0):
    """
    Baseline function used in Kennedy (2020), based on Model from Gyorfi et al (2006)

    Parameters
    ----------
    X: array-like
        input data to use
    dim: int, default 0
        Dimension of X to use

    Returns
    -------
    A value for each row of X
    """
    X = _get_values_only(X)
    return (X[:, dim] <= -.5) * 0.5 * (X[:, dim] + 2) ** 2 + (X[:, dim] / 2 + 0.875) * (
            (X[:, dim] > -1 / 2) & (X[:, dim] < 0)) + ((X[:, dim] > 0) & (X[:, dim] < .5)) \
             * (-5 * (X[:, dim] - 0.2) ** 2 + 1.075) + (X[:, dim] > .5) * (
                   X[:, dim] + 0.125)


def binary_gyorfi_baseline(X, dim: int = 0):
    """
    Adaptation of gyorfi baseline for binary data (scaled between 0 and 1)

    Parameters
    ----------
    X: array-like
        input data to use
    dim: int, default 0
        Dimension of X to use

    Returns
    -------
    A value for each row of X
    """
    return gyorfi_baseline(X, dim=dim) / 1.5


def baseline_wa(X, dim: int = 2):
    """
    Baseline function from Wager & Athey (2018)

    Parameters
    ----------
    X: array-like
        input data to use
    dim: int, default 2
        Dimension of X to use
    """
    X = _get_values_only(X)
    return 2*X[:, dim] - 1


# Error models --------------------------------------------------------------------
def normal_error_model(X, sd: float = 1):
    """
    Generate errors according to N(0, sd)
    Parameters
    ----------
    X: array-like
        input data to use
    sd: float, default 1
        standard deviation to use

    Returns
    -------
    An error value for each row of X
    """
    return np.random.normal(0, sd, X.shape[0])


def cos_error_model(X, dim: int = 0):
    """
    Generate errors according to the error model used in Kennedy (2020)
    Parameters
    ----------
    X: array-like
        input data to use
    dim: int, default 0
        Dimension of X to use

    Returns
    -------
    An error value for each row of X
    """
    X = _get_values_only(X)
    return np.random.normal(0, 0.2 - 0.1 * np.cos(2 * np.pi * X[:, dim]), X.shape[0])


# Covariate models --------------------------------------------------------------------
def normal_covariate_model(n: int, d: int = 1, sd: float = 1):
    """
    Generate normal covariates

    Parameters
    ----------
    n: int
        number of observations to generate
    d: int
        number of dimensions
    sd: float
        standard deviation

    Returns
    -------
    np. array (n x d) with generated covariates
    """
    return np.random.normal(0, sd, d * n).reshape(-1, d)


def uniform_covariate_model(n: int, d: int = 1, low: float = -1, high: float = 1):
    """
    Generate uniform covariates

    Parameters
    ----------
    n: int
        number of observations to generate
    d: int
        number of dimensions
    low: float
        lower bound of hypercube
    high: float
        upper bound of hypercube

    Returns
    -------
    np. array (n x d) with generated covariates
    """
    return np.random.uniform(low=low, high=high, size=d * n).reshape(-1, d)


# helper functions ----------------------------------------------------------
def _get_values_only(X):
    # wrapper to return only values of data frame
    if isinstance(X, pd.DataFrame):
        X = X.values
    return X


def _check_is_callable(input, name: str = ''):
    if callable(input):
        pass
    else:
        raise ValueError('Input {} needs to be a callable function so it can '
                         'be used to create simulation.'.format(name))


class ModelCaller:
    """
    Class to act as a wrapper around any function if arguments should be changed flexibly
    """
    def __init__(self, model, args: dict = None):
        """
        Set model and arguments to call with
        """
        _check_is_callable(model, 'model')
        self.model = model
        if args is None:
            args = {}
        self.args = args

    def __call__(self, **call_args):
        if call_args is None:
            call_args = {}
        return self.model(**call_args, **self.args)


# Constants ---------------------------------------------------------------
BASE_COVARIATE_MODEL = uniform_covariate_model
BASE_BASELINE_MODEL = gyorfi_baseline
BASE_ERROR_MODEL = cos_error_model


