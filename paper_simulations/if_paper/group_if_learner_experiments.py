"""
Author: Alicia Curth
Utils to create the Group-IF-learner simulation results (Simulation study 1.2) in Curth, Alaa and
van der Schaar (2020)
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from iflearn.utils.base import clone
from iflearn.simulation_utils.base import BASE_BASELINE_MODEL
from iflearn.simulation_utils.treatment_effects import make_te_data

from iflearn.treatment_effects.base import CATE_NAME, RR_NAME
from iflearn.treatment_effects.oracle_scoring import fit_and_score_te_oracle
from iflearn.treatment_effects.group_learner import GroupIFLearner

N_TEST_BASE = 1000
N_REPEATS_BASE = 500
MSE_NAME = 'neg_mean_squared_error'
METHOD_NAMES = {'t': 'Plugin-unadjusted',
                'adjt': 'Plugin-adjusted',
                'if': 'IF-unadjusted',
                'adjif': 'IF-adjusted',
                'cher': 'Chernozhukov et al (2018)'}


def _eval_one_group_setting(te_estimator, train, test, n_train, n_test, d=1,
                            te_function=None, baseline_estimator=None, binary_y=False,
                            baseline_model=BASE_BASELINE_MODEL, setting=CATE_NAME,
                            covariate_model=None,
                            propensity_model=None, error_model=None,
                            selection_bias=None, seedy=42):
    np.random.seed(seedy)
    X, y, w, t, p, _ = make_te_data(n=n_train + n_test, d=d, te_model=te_function,
                                    baseline_model=baseline_model, covariate_model=covariate_model,
                                    propensity_model=propensity_model, binary_y=binary_y,
                                    error_model=error_model, setting=setting,
                                    seedy=seedy, selection_bias=selection_bias)

    if setting == CATE_NAME:
        cherscore = - fit_and_score_te_oracle(GroupIFLearner(te_estimator=te_estimator,
                                                             base_estimator=baseline_estimator,
                                                             setting=setting,
                                                             plug_in_corrected=False,
                                                             baseline_adjustment=True,
                                                             efficient_est=True),
                                              X, y, w, p, t,
                                              train=train,
                                              test=test,
                                              scorer=MSE_NAME,
                                              return_test_score_only=True)
    else:
        cherscore = np.nan

    tscore = - fit_and_score_te_oracle(GroupIFLearner(te_estimator=te_estimator,
                                                      base_estimator=baseline_estimator,
                                                      setting=setting,
                                                      plug_in_corrected=False,
                                                      baseline_adjustment=False),
                                       X, y, w, p, t,
                                       train=train,
                                       test=test,
                                       scorer=MSE_NAME,
                                       return_test_score_only=True)
    tadjscore = - fit_and_score_te_oracle(GroupIFLearner(te_estimator=te_estimator,
                                                         base_estimator=baseline_estimator,
                                                         setting=setting,
                                                         plug_in_corrected=False,
                                                         baseline_adjustment=True),
                                          X, y, w, p, t,
                                          train=train,
                                          test=test,
                                          scorer=MSE_NAME,
                                          return_test_score_only=True)
    ifscore = - fit_and_score_te_oracle(GroupIFLearner(te_estimator=te_estimator,
                                                       base_estimator=baseline_estimator,
                                                       setting=setting,
                                                       plug_in_corrected=True,
                                                       baseline_adjustment=False),
                                        X, y, w, p, t,
                                        train=train,
                                        test=test,
                                        scorer=MSE_NAME,
                                        return_test_score_only=True)

    ifscoreadj = - fit_and_score_te_oracle(GroupIFLearner(te_estimator=te_estimator,
                                                          base_estimator=baseline_estimator,
                                                          setting=setting,
                                                          plug_in_corrected=True,
                                                          baseline_adjustment=True),
                                           X, y, w, p, t,
                                           train=train,
                                           test=test,
                                           scorer=MSE_NAME,
                                           return_test_score_only=True)

    scores = tscore, tadjscore, ifscore, ifscoreadj, cherscore

    for score in scores:
        # remove round if very bad case of boundary bias
        if score > n_test:
            return np.nan, np.nan, np.nan, np.nan, np.nan

    return scores


def eval_setting_group_repeat(te_estimator, n_train, baseline_estimator=None, repeats=1000,
                              n_test=1000, d=1, setting=CATE_NAME,
                              te_function=None, covariate_model=None, binary_y=False,
                              baseline_model=None, selection_bias=None,
                              propensity_model=None,
                              error_model=None,
                              pre_dispatch='2*n_jobs', n_jobs=1, verbose=0):
    train = [i for i in range(n_train)]
    test = [i for i in range(n_train, n_train + n_test)]

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    scores = parallel(
        delayed(_eval_one_group_setting)(clone(te_estimator), train, test, n_train, n_test,
                                         d=d, selection_bias=selection_bias,
                                         baseline_estimator=baseline_estimator,
                                         te_function=te_function, setting=setting,
                                         covariate_model=covariate_model,
                                         baseline_model=baseline_model, binary_y=binary_y,
                                         propensity_model=propensity_model,
                                         error_model=error_model,
                                         seedy=i)
        for i in range(repeats))

    zipped_scores = list(zip(*scores))
    tscores = zipped_scores.pop(0)
    adjtscores = zipped_scores.pop(0)
    ifscores = zipped_scores.pop(0)
    adjifscores = zipped_scores.pop(0)
    if setting == 'CATE':
        cherscores = zipped_scores.pop(0)
        return tscores, adjtscores, ifscores, adjifscores, cherscores
    else:
        return tscores, adjtscores, ifscores, adjifscores


def eval_range_n_group(te_estimator, range_n, repeats=1000, n_test=1000, d=1,
                       baseline_estimator=None,
                       te_function=None, setting=CATE_NAME,
                       baseline_model=None,
                       propensity_model=None,
                       error_model=None,
                       pre_dispatch='2*n_jobs', n_jobs=1,
                       verbose=0):
    if setting == CATE_NAME:
        resultframe = pd.DataFrame(columns=['t_mean', 'adjt_mean', 'if_mean', 'adjif_mean',
                                            'cher_mean', 't_sd',
                                            'adjt_sd', 'if_sd', 'adjif_sd', 'cher_sd'])
    else:
        resultframe = pd.DataFrame(columns=['t_mean', 'adjt_mean', 'if_mean', 'adjif_mean',
                                            't_sd', 'adjt_sd', 'if_sd', 'adjif_sd'])

    for n in range_n:
        print('number of train-samples: {}'.format(n))
        scores = eval_setting_group_repeat(te_estimator=te_estimator, n_train=n,
                                           repeats=repeats, n_test=n_test,
                                           setting=setting,
                                           baseline_estimator=baseline_estimator,
                                           d=d, te_function=te_function,
                                           baseline_model=baseline_model,
                                           propensity_model=propensity_model,
                                           error_model=error_model,
                                           pre_dispatch=pre_dispatch,
                                           n_jobs=n_jobs,
                                           verbose=verbose)

        # need possiblity for nans in case something goes wrong in small n regime
        means = np.array([np.nanmean(np.array(x)) for x in scores])
        sds = np.array([np.nanstd(np.array(x)) / np.sqrt(sum(~np.isnan(x))) for x in scores])
        resultframe.loc[n, :] = np.concatenate((means, sds))

    return resultframe
