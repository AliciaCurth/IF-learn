"""
Author: Alicia Curth
Some tests for iflearn.simulation_utils.base and iflearn.simulation.utils.treatment_effects
"""
import pytest
import pandas as pd
import numpy as np

from iflearn.simulation_utils.base import ModelCaller, _check_is_callable, _get_values_only, \
                                          uniform_covariate_model


def test__check_is_callable():
    # check that it indeed considers models callable
    _check_is_callable(uniform_covariate_model)

    # test that noncallable objects are caught
    notcallable = pd.DataFrame(np.array([0, 1]))
    with pytest.raises(ValueError):
        _check_is_callable(notcallable, 'noncallable object')


def test__get_values_only():
    # test that df gets correctly processed
    df = pd.DataFrame(np.array([0, 1]))
    np.testing.assert_almost_equal(_get_values_only(df), np.array([0, 1]).reshape(2,1))

    # check that nothing happens if input is np.array
    assert (np.array([0, 1]) == _get_values_only(np.array([0, 1]))).all()


def test_modelcaller():
    # check that constructor fails if model not callable
    with pytest.raises(ValueError):
        ModelCaller('notafunction')

    # check that the caller does the same as the original function
    # in different setups
    np.random.seed(42)
    X = uniform_covariate_model(n=2, d=2)

    caller_1 = ModelCaller(uniform_covariate_model, args={'n': 2, 'd': 2})
    np.random.seed(42)
    assert (X == caller_1()).all()

    caller_2 = ModelCaller(uniform_covariate_model, args={'n': 2})
    np.random.seed(42)
    assert (X == caller_2(d=2)).all()

    # check we can add lower and higher as we want
    np.random.seed(42)
    X = uniform_covariate_model(n=2, d=2, high=1, low=0)

    caller_3 = ModelCaller(uniform_covariate_model, args={'high': 1, 'low': 0})
    np.random.seed(42)
    assert (X == caller_3(n=2, d=2)).all()

