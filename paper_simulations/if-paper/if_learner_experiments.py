"""
Author: Alicia Curth
Utils to create the IF-learner simulation results in Curth, Alaa and van der Schaar (2020)
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from iflearn.utils.base import clone
from iflearn.simulation_utils.base import BASE_BASELINE_MODEL
from iflearn.simulation_utils.treatment_effects import make_te_data, BASE_TE_MODEL

from iflearn.treatment_effects.base import CATE_NAME, RR_NAME
from iflearn.treatment_effects.oracle_scoring import fit_and_score_te_oracle
from iflearn.treatment_effects.base_learners import IFLearnerTE, IFTEOracle, PlugInTELearner, \
    TEOracle

METHOD_NAMES = {'t': 'T-learner',
                'ot': 'T-Oracle',
                'if': 'IF-learner',
                'oif': 'Oracle IF-learner'}
N_TRAIN_BASE = 500
N_TEST_BASE = 1000
N_REPEATS_BASE = 500


def _eval_one_setting(base_estimator, train, test, n_train, n_test, d=1,
                      te_function=None, baseline_model=None,
                      te_estimator=None, setting=CATE_NAME,
                      propensity_model=None, covariate_model=None,
                      error_model=None, binary_y=False,
                      selection_bias=None,
                      seedy=42):
    np.random.seed(seedy)
    X, y, w, t, p = make_te_data(n=n_train + n_test, d=d, te_model=te_function,
                                 baseline_model=baseline_model, covariate_model=covariate_model,
                                 propensity_model=propensity_model, binary_y=binary_y,
                                 error_model=error_model,
                                 seedy=seedy, selection_bias=selection_bias)

    # update some settings for safety
    if binary_y:
        setting = RR_NAME
    if te_function is None:
        te_function = BASE_TE_MODEL
    if baseline_model is None:
        baseline_model = BASE_BASELINE_MODEL

    if binary_y:
        tscore = - fit_and_score_te_oracle(PlugInTELearner(base_estimator, setting=setting),
                                           X, y, w, p, t,
                                           train=train,
                                           test=test,
                                           scorer='neg_mean_squared_error',
                                           return_test_score_only=True)
        toracle = TEOracle(te_model=te_function,
                           base_model=baseline_model)
        otscore = - fit_and_score_te_oracle(toracle,
                                            X, y, w, p, t,
                                            train=train,
                                            test=test,
                                            scorer='neg_mean_squared_error',
                                            return_test_score_only=True)

        ifscore = - fit_and_score_te_oracle(IFLearnerTE(base_estimator=base_estimator,
                                                        setting=setting,
                                                        te_estimator=te_estimator),
                                            X, y, w, p, t,
                                            train=train,
                                            test=test,
                                            scorer='neg_mean_squared_error',
                                            return_test_score_only=True)

        oifscore = - fit_and_score_te_oracle(IFTEOracle(te_estimator, te_model=te_function,
                                                        base_model=baseline_model,
                                                        setting=setting),
                                             X, y, w, p, t,
                                             train=train,
                                             test=test,
                                             scorer='neg_mean_squared_error',
                                             return_test_score_only=True)

    else:
        tscore = - fit_and_score_te_oracle(PlugInTELearner(base_estimator), X, y, w, p, t,
                                           train=train,
                                           test=test,
                                           scorer='neg_mean_squared_error',
                                           return_test_score_only=True)
        otscore = - fit_and_score_te_oracle(TEOracle(te_model=te_function,
                                                     base_model=baseline_model),
                                            X, y, w, p, t,
                                            train=train,
                                            test=test,
                                            scorer='neg_mean_squared_error',
                                            return_test_score_only=True)

        ifscore = - fit_and_score_te_oracle(IFLearnerTE(base_estimator), X, y, w, p, t,
                                            train=train,
                                            test=test,
                                            scorer='neg_mean_squared_error',
                                            return_test_score_only=True)

        oifscore = - fit_and_score_te_oracle(IFTEOracle(base_estimator, te_model=te_function,
                                                        base_model=baseline_model),
                                             X, y, w, p, t,
                                             train=train,
                                             test=test,
                                             scorer='neg_mean_squared_error',
                                             return_test_score_only=True)
    scores = tscore, otscore, ifscore, oifscore
    for score in scores:
        # remove round if very bad case of boundary bias
        if score > n_test:
            return np.nan, np.nan, np.nan, np.nan
    return scores


def eval_setting_repeat(estimator, n_train, repeats=N_REPEATS_BASE, n_test=N_TEST_BASE, d=1,
                        te_function=None, baseline_model=None, selection_bias=None, binary=False,
                        te_estimator=None,
                        propensity_model=None, error_model=None,
                        pre_dispatch='2*n_jobs', n_jobs=1, verbose=0):
    train = [i for i in range(n_train)]
    test = [i for i in range(n_train, n_train + n_test)]

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    # dispatch with different seeds
    scores = parallel(delayed(_eval_one_setting)(clone(estimator), train, test, n_train, n_test,
                                                 d=d, selection_bias=selection_bias,
                                                 te_function=te_function, binary=binary,
                                                 baseline_model=baseline_model,
                                                 te_estimator=te_estimator,
                                                 propensity_model=propensity_model,
                                                 error_model=error_model,
                                                 seedy=i)
                      for i in range(repeats))

    zipped_scores = list(zip(*scores))
    tscores = zipped_scores.pop(0)
    otscores = zipped_scores.pop(0)
    ifscores = zipped_scores.pop(0)
    oifscores = zipped_scores.pop(0)
    return tscores, otscores, ifscores, oifscores


def eval_range_n(estimator, range_n, repeats=N_REPEATS_BASE, n_test=N_TEST_BASE, d=1,
                 te_function=None, baseline_model=None, binary=False, te_estimator=None,
                 propensity_model=None, error_model=None,
                 pre_dispatch='2*n_jobs', n_jobs=1, verbose=0):
    # evaluate performance on range of n
    resultframe = pd.DataFrame(columns=['t_mean', 'ot_mean', 'if_mean', 'oif_mean', 't_sd',
                                        'ot_sd', 'if_sd', 'oif_sd'])

    for n in range_n:
        print('number of train-samples: {}'.format(n))
        scores = eval_setting_repeat(estimator=estimator, n_train=n, te_estimator=te_estimator,
                                     repeats=repeats, n_test=n_test,
                                     d=d, te_function=te_function,
                                     baseline_model=baseline_model,
                                     propensity_model=propensity_model,
                                     error_model=error_model,
                                     pre_dispatch=pre_dispatch,
                                     n_jobs=n_jobs,
                                     verbose=verbose, binary=binary
                                     )
        # need possiblity for nans in case something goes wrong in small n regime
        means = np.array([np.nanmean(np.array(x)) for x in scores])
        sds = np.array([np.nanstd(np.array(x)) / np.sqrt(repeats) for x in scores])
        resultframe.loc[n, :] = np.concatenate((means, sds))

    return resultframe


def eval_range_d(estimator, range_d, propensity_model=None,
                 repeats=N_REPEATS_BASE, binary=False, te_estimator=None,
                 n_test=N_TEST_BASE, n_train=N_TRAIN_BASE,
                 te_function=None, baseline_model=None, error_model=None,
                 pre_dispatch='2*n_jobs', n_jobs=1, verbose=0):
    resultframe = pd.DataFrame(columns=['t_mean', 'ot_mean', 'if_mean', 'oif_mean', 't_sd',
                                        'ot_sd', 'if_sd', 'oif_sd'])
    for d in range_d:
        print('Ambient dimension: {}'.format(d))
        scores = eval_setting_repeat(estimator=estimator, n_train=n_train,
                                     repeats=repeats, n_test=n_test, te_estimator=te_estimator,
                                     d=d, te_function=te_function,
                                     baseline_model=baseline_model, binary=binary,
                                     propensity_model=propensity_model,
                                     error_model=error_model,
                                     pre_dispatch=pre_dispatch,
                                     n_jobs=n_jobs,
                                     verbose=verbose)
        # need possiblity for nans in case something goes wrong in small n regime
        means = [np.nanmean(np.array(x)) for x in scores]
        sds = [np.nanstd(np.array(x)) / np.sqrt(repeats) for x in scores]
        resultframe.loc[d, :] = means + sds

    return resultframe


def eval_range_bias(estimator, range_p, propensity_class, repeats=N_REPEATS_BASE,
                    n_test=N_TEST_BASE, n_train=N_TRAIN_BASE,
                    d=1, te_function=None, binary=False, te_estimator=None,
                    baseline_model=None, error_model=None,
                    pre_dispatch='2*n_jobs', n_jobs=1, verbose=0):
    resultframe = pd.DataFrame(columns=['t_mean', 'ot_mean', 'if_mean', 'oif_mean', 't_sd',
                                        'ot_sd', 'if_sd', 'oif_sd'])
    for p in range_p:
        selection_model = propensity_class(p)
        print('Bias with parameter: {}'.format(p))
        scores = eval_setting_repeat(estimator=estimator, n_train=n_train,
                                     repeats=repeats, n_test=n_test, te_estimator=te_estimator,
                                     d=d, te_function=te_function, binary=binary,
                                     baseline_model=baseline_model,
                                     propensity_model=None,
                                     selection_bias=selection_model,
                                     error_model=error_model,
                                     pre_dispatch=pre_dispatch,
                                     n_jobs=n_jobs,
                                     verbose=verbose)
        # need possiblity for nans in case something goes wrong in small n regime
        means = [np.nanmean(np.array(x)) for x in scores]
        sds = [np.nanstd(np.array(x)) / np.sqrt(repeats) for x in scores]
        resultframe.loc[p, :] = means + sds

    return resultframe


def make_plot_frame(results, methods=METHOD_NAMES, dim_name='n'):
    plot_frame = pd.DataFrame(columns=[dim_name, 'method', 'meanmse', 'sd'])
    for key, val in methods.items():
        new_frame = pd.DataFrame(data={dim_name: results.index.values,
                                       'method': val,
                                       'meanmse': results[key + '_mean']})

        plot_frame = pd.concat([plot_frame, new_frame])

    convert_dict = {dim_name: float, 'meanmse': float}
    plot_frame['meanmse'] = plot_frame['meanmse'] * 1000

    return plot_frame.astype(convert_dict)
