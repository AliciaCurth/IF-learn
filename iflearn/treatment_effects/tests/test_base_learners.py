"""
Author: Alicia Curth
Unittests for the base_learners module
"""
import pytest
from iflearn.treatment_effects.base import _get_po_function, _get_te_eif, \
    eif_transformation_CATE, eif_transformation_RR, po_function_CATE, po_function_RR, CATE_NAME, \
    RR_NAME, ht_te_transformation
from iflearn.treatment_effects.base_learners import BaseTEModel, PlugInTELearner, IFLearnerTE, \
    TEOracle, IFTEOracle
from iflearn.simulation_utils.treatment_effects import make_te_data, BASE_TE_MODEL, \
    BASE_BASELINE_MODEL
from pygam import GAM

import pandas as pd
import numpy as np


# Tests for base functions ----------------------------------------------------------
def test_settingscheck_eif_po():
    # test that the setting checker works
    notcallable = pd.DataFrame(np.array([0, 1]))
    with pytest.raises(ValueError):
        _get_te_eif('nonsense')

    with pytest.raises(ValueError):
        _get_te_eif(notcallable)

    with pytest.raises(ValueError):
        _get_po_function('nonsense')

    with pytest.raises(ValueError):
        _get_po_function(notcallable)

    assert _get_po_function(RR_NAME) == po_function_RR
    assert _get_po_function(CATE_NAME) == po_function_CATE
    assert _get_te_eif(RR_NAME) == eif_transformation_RR
    assert _get_te_eif(CATE_NAME) == eif_transformation_CATE

    # test that we can pass down a function
    po_func = _get_po_function(po_function_RR)
    eif_func = _get_te_eif(eif_transformation_RR)

    # test that warning is raised
    with pytest.warns(UserWarning):
        _get_po_function(RR_NAME, binary=False)

    with pytest.warns(UserWarning):
        _get_te_eif(RR_NAME, binary=False)


def test_po_functions():
    # test that the potential outcome functions give right output
    mu_0 = np.array([1, 2, 3])
    mu_1 = np.array([2, 2, 2])
    rr_true = np.array([2, 1, 2 / 3])
    cate_true = np.array([1, 0, -1])

    cate_test = po_function_CATE(mu_0, mu_1)
    np.testing.assert_almost_equal(cate_test, cate_true)
    rr_test = po_function_RR(mu_0, mu_1)
    np.testing.assert_almost_equal(rr_test, rr_true)


def test_eifs():
    # test that eifs give correct output
    mu_0 = np.array([1, 2])
    mu_1 = np.array([2, 4])
    y = np.array([2, 2])
    w = np.array([1, 0])
    p = np.array([0.5, 0.5])
    w_1 = w / p
    w_0 = (1 - w) / (1 - p)

    # eif for cate
    eif_cate_true = np.array(
        [2 / 0.5 + (1 - 1 / 0.5) * 2 - 1, -2 * 1 / 0.5 + 1 * 4 - (1 - 1 / 0.5) * 2])
    eif_cate_test = eif_transformation_CATE(y, w, p, mu_0, mu_1)
    np.testing.assert_almost_equal(eif_cate_true, eif_cate_test)

    # eif for rr
    eif_rr_true = 1 / mu_0 * (w_1 * y + (1 - w_1) * mu_1 - mu_1) - mu_1 / (mu_0 ** 2) * (
                w_0 * y + (1 - w_0) * mu_0
                - mu_0) + mu_1 / mu_0
    eif_rr_test = eif_transformation_RR(y, w, p, mu_0, mu_1)
    np.testing.assert_almost_equal(eif_rr_test, eif_rr_true)

    # test that error is raised if there is a zero in denominator
    with pytest.raises(ValueError):
        eif_transformation_RR(y, w, p, np.array([0, 0]), mu_1)

    # horvitz thompson transformation
    ht_true = np.array([2 / 0.5, -2 / 0.5])
    ht_test = ht_te_transformation(y, w, p)
    np.testing.assert_almost_equal(ht_test, ht_true)


# tests for learners ----------------------------------------------------------------
def test_model_constructors():
    # test that the right errors are thrown because cannot be constructed
    with pytest.raises(TypeError):
        BaseTEModel()

    with pytest.raises(ValueError):
        IFLearnerTE(None)


def test_te_oracles():
    # get data without noise
    X, y, w, ite, p, bs = make_te_data(noise=False)

    # test true oracle
    oracle = TEOracle(te_model=BASE_TE_MODEL, base_model=BASE_BASELINE_MODEL)
    oracle.fit(X, y, w, p)
    te, mu_0, mu_1 = oracle.predict(X, return_po=True)
    np.testing.assert_almost_equal(te, ite)
    np.testing.assert_almost_equal(mu_0, bs)
    np.testing.assert_almost_equal(mu_1, ite+bs)
    np.testing.assert_almost_equal(oracle.predict(X), ite)

    # test if oracle
    oracle = IFTEOracle(te_estimator=GAM(),
                        te_model=BASE_TE_MODEL, base_model=BASE_BASELINE_MODEL)
    oracle.fit(X, y, w, p)
    te, mu_0, mu_1 = oracle.predict(X, return_po=True)
    np.testing.assert_almost_equal(te, ite)
    np.testing.assert_almost_equal(mu_0, bs)
    np.testing.assert_almost_equal(mu_1, ite+bs)
    np.testing.assert_almost_equal(oracle.predict(X), ite)







if __name__ == '__main__':
    test_te_oracles()
