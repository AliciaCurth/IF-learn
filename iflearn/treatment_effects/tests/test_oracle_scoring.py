"""
Author: Alicia Curth
Tests for iflearn.treatment_effects.oracle_scoring
"""
import pytest
import numpy as np
import math

from sklearn.utils import _safe_indexing
from sklearn.metrics import mean_squared_error

from iflearn.simulation_utils.treatment_effects import make_te_data
from iflearn.treatment_effects.base_learners import IFLearnerTE
from iflearn.treatment_effects.oracle_scoring import fit_and_score_te_oracle

from pygam import LinearGAM, LogisticGAM


def test_exceptions():
    # get data
    X, y, w, ite, p, bs = make_te_data(n=200)
    train = [i for i in range(100)]
    test = [i for i in range(100, 200)]

    with pytest.raises(ValueError):
        # pass incorrect type of estimator
        fit_and_score_te_oracle(LinearGAM(), X, y, w, p, ite,
                                train=train,
                                test=test,
                                scorer='neg_mean_squared_error',
                                return_test_score_only=True)

    with pytest.raises(ValueError):
        # fit should throw an error
        fit_and_score_te_oracle(IFLearnerTE(LogisticGAM()), X, y, w, p, ite,
                                train=train,
                                test=test,
                                scorer='neg_mean_squared_error',
                                return_test_score_only=True, error_score='raise')

    with pytest.raises(ValueError):
        # fit should throw an error because error score is incorrect
        fit_and_score_te_oracle(IFLearnerTE(LogisticGAM()), X, y, w, p, ite,
                                train=train,
                                test=test,
                                scorer='neg_mean_squared_error',
                                return_test_score_only=False, error_score='asdfad')

    # assert we get error score otherwise
    score = fit_and_score_te_oracle(IFLearnerTE(LogisticGAM()), X, y, w, p, ite,
                                    train=train,
                                    test=test,
                                    scorer='neg_mean_squared_error',
                                    return_test_score_only=True, error_score=np.nan)
    assert math.isnan(score)


def test_scores():
    # get data
    X, y, w, ite, p, bs = make_te_data(n=200)
    train = [i for i in range(100)]
    test = [i for i in range(100, 200)]

    # test that score is correct by pre-training IFLearner outside of scorer
    # split data
    X_train, y_train, w_train, p_train = _safe_indexing(X, train), _safe_indexing(y, train), \
                                         _safe_indexing(w, train), _safe_indexing(p, train)
    X_test, t_test = _safe_indexing(X, test),  _safe_indexing(ite, test)

    # fit if-learner and get predictions on test set
    if_learner = IFLearnerTE(LinearGAM())
    if_learner.fit(X_train, y_train, w_train, p_train)
    t_pred = if_learner.predict(X_test)
    neg_mse = - mean_squared_error(t_test, t_pred)

    # score output
    score = fit_and_score_te_oracle(IFLearnerTE(LinearGAM()), X, y, w, p, ite,
                                    train=train,
                                    test=test,
                                    scorer='neg_mean_squared_error',
                                    return_test_score_only=True, error_score=np.nan)

    np.testing.assert_almost_equal(score, neg_mse)

    # smoke test some other capabilities
    # test that we can pass parameters too
    score = fit_and_score_te_oracle(IFLearnerTE(LinearGAM()), X, y, w, p, ite,
                                    train=train,
                                    test=test, parameters={'te_estimator': LinearGAM()},
                                    scorer='neg_mean_squared_error',
                                    return_test_score_only=True, error_score=np.nan)
    np.testing.assert_almost_equal(score, neg_mse)
