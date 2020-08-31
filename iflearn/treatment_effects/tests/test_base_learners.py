"""
Author: Alicia Curth
Tests for iflearn.treatment_effects.base_learners
"""
import pytest
import numpy as np

from iflearn.treatment_effects.base import eif_transformation_CATE, RR_NAME
from iflearn.treatment_effects.base_learners import BaseTEModel, PlugInTELearner, IFLearnerTE, \
    TEOracle, IFTEOracle
from iflearn.simulation_utils.base import binary_gyorfi_baseline
from iflearn.simulation_utils.treatment_effects import make_te_data, BASE_TE_MODEL, \
    BASE_BASELINE_MODEL

from sklearn.model_selection import StratifiedKFold
from pygam import LinearGAM, LogisticGAM


# helper function
def get_surrogate_predictions(X, y, w, pred_mask=None):
    if pred_mask is None:
        pred_mask = np.ones(len(y), dtype=bool)
        fit_mask = pred_mask
    else:
        fit_mask = ~pred_mask
    # get surrogates
    model_1 = LinearGAM()
    model_1.fit(X[fit_mask & (w == 1), :], y[fit_mask & (w == 1)])
    mu_1_plug = model_1.predict(X[pred_mask, :])

    model_0 = LinearGAM()
    model_0.fit(X[fit_mask & (w == 0), :], y[fit_mask & (w == 0)])
    mu_0_plug = model_0.predict(X[pred_mask, :])

    return mu_0_plug, mu_1_plug


def test_model_constructors():
    # test that the right errors are thrown because cannot be constructed
    with pytest.raises(TypeError):
        BaseTEModel()

    with pytest.raises(ValueError):
        IFLearnerTE(None)

    # test other configurations of base learners
    if_learner1 = IFLearnerTE(None, base_estimator=LinearGAM())
    if_learner2 = IFLearnerTE(te_estimator=LinearGAM(), base_estimator=None)


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
    oracle = IFTEOracle(te_estimator=LinearGAM(),
                        te_model=BASE_TE_MODEL, base_model=BASE_BASELINE_MODEL)
    oracle.fit(X, y, w, p)
    te, mu_0, mu_1 = oracle.predict(X, return_po=True)
    np.testing.assert_almost_equal(te, ite)
    np.testing.assert_almost_equal(mu_0, bs)
    np.testing.assert_almost_equal(mu_1, ite+bs)
    np.testing.assert_almost_equal(oracle.predict(X), ite)


def test_plugin_learner():
    # get data without noise
    X, y, w, ite, p, bs = make_te_data(n=200, noise=False)

    # get surrogates
    mu_0_plug, mu_1_plug = get_surrogate_predictions(X, y, w)

    # use plug in learner
    p_model = PlugInTELearner(LinearGAM())
    p_model.fit(X, y, w, p)
    te, mu_0, mu_1 = p_model.predict(X, return_po=True)

    # test outcomes
    np.testing.assert_almost_equal(te, mu_1_plug - mu_0_plug)
    np.testing.assert_almost_equal(mu_0, mu_0_plug)
    np.testing.assert_almost_equal(mu_1, mu_1_plug)
    np.testing.assert_almost_equal(p_model.predict(X), mu_1_plug - mu_0_plug)

    # check that binary setting also works (smoketest)
    X, y, w, ite, p, bs = make_te_data(n=200, baseline_model=binary_gyorfi_baseline,
                                       noise=False,  binary=True)
    p_model = PlugInTELearner(LogisticGAM(), binary=True, setting=RR_NAME)
    p_model.fit(X, y, w, p)
    te, mu_0, mu_1 = p_model.predict(X, return_po=True)


def test_if_learner():
    # get data without noise
    X, y, w, ite, p, bs = make_te_data(n=200, noise=False)

    # get surrogate predictions to compare against po predictions
    mu_0_plug, mu_1_plug = get_surrogate_predictions(X, y, w)

    # get surrogate predictions for two folds as inside the iflearner
    splitter = StratifiedKFold(n_splits=2, shuffle=True,
                               random_state=42)
    idx_list = []
    for train_index, test_index in splitter.split(X, w):
        idx_list.append((train_index, test_index))

    fold2_mask = np.zeros(200, dtype=bool)
    fold2_mask[idx_list[0][1]] = 1
    mu_0, mu_1 = np.zeros(200), np.zeros(200)
    mu_0[~fold2_mask], mu_1[~fold2_mask] = get_surrogate_predictions(X, y, w, pred_mask=~fold2_mask)
    mu_0[fold2_mask], mu_1[fold2_mask] = get_surrogate_predictions(X, y, w, pred_mask=fold2_mask)
    pseudo_outcome = eif_transformation_CATE(y, w, p, mu_0, mu_1)

    # make second stage model
    t_model = LinearGAM()
    t_model.fit(X, pseudo_outcome)
    te_debiased = t_model.predict(X)

    # fit if learner
    if_learner = IFLearnerTE(LinearGAM(), n_folds=2, random_state=42, fit_base_model=True)
    if_learner.fit(X, y, w, p)
    te, mu_0, mu_1 = if_learner.predict(X, return_po=True)

    # test outcomes
    np.testing.assert_almost_equal(te, te_debiased)
    np.testing.assert_almost_equal(mu_0, mu_0_plug)
    np.testing.assert_almost_equal(mu_1, mu_1_plug)
    np.testing.assert_almost_equal(if_learner.predict(X), te_debiased)

    with pytest.raises(ValueError):
        # predicting po when base model not fitted should not be possible
        if_learner = IFLearnerTE(LinearGAM(), n_folds=2, random_state=42)
        if_learner.fit(X, y, w, p)
        te, mu_0, mu_1 = if_learner.predict(X, return_po=True)

    with pytest.warns(UserWarning):
        # warning raised if only one fold?
        if_learner = IFLearnerTE(LinearGAM(), n_folds=1, random_state=42)
        if_learner.fit(X, y, w, p)

    # check that binary setting also works (smoketest)
    X, y, w, ite, p, bs = make_te_data(n=200, baseline_model=binary_gyorfi_baseline,
                                       noise=False,  binary=True)
    if_learner = IFLearnerTE(base_estimator=LogisticGAM(), te_estimator=LinearGAM(),
                             binary=True, setting=RR_NAME, fit_base_model=True)
    if_learner.fit(X, y, w, p)
    te, mu_0, mu_1 = if_learner.predict(X, return_po=True)
