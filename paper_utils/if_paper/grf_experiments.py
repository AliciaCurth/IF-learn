"""
Author: Alicia Curth
Utils to create the GRF simulation results in Curth, Alaa and van der Schaar (2020)
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

from iflearn.simulation_utils.treatment_effects import make_te_data
from iflearn.treatment_effects.oracle_scoring import _safe_split_te

from paper_utils.if_paper.if_learner_experiments import N_REPEATS_BASE, N_TRAIN_BASE, \
    N_TEST_BASE

# rpy2 needs pandas < 1.0.0 due to deprecation of pd.from_items()
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

NUM_TREES_BASE = 2000
PAPER_UTILS_ROOT = 'paper_utils/if_paper/'


def _eval_one_setting_grf(train, test, n_train, n_test, num_trees=NUM_TREES_BASE, d=1,
                          te_function=None, baseline_model=None,
                          propensity_model=None, covariate_model=None,
                          error_model=None, binary_y=False,
                          selection_bias=None,
                          seedy=42, root=PAPER_UTILS_ROOT):
    # get data
    np.random.seed(seedy)
    X, y, w, t, p, _ = make_te_data(n=n_train + n_test, d=d, te_model=te_function,
                                    baseline_model=baseline_model, covariate_model=covariate_model,
                                    propensity_model=propensity_model, binary_y=binary_y,
                                    error_model=error_model,
                                    seedy=seedy, selection_bias=selection_bias)

    # split data
    X_train, y_train, w_train, p_train, _ = _safe_split_te(X, y, w, p, t, train)
    X_test, _, _, _, t_test = _safe_split_te(X, y, w, p, t, test)

    # convert to R objects
    r_y = robjects.FloatVector(y_train)
    r_x = robjects.r.matrix(X_train, n_train, d)
    r_w = robjects.IntVector(w_train)
    r_p = robjects.FloatVector(p_train)
    r_x_test = robjects.r.matrix(X_test, n_test, d)

    # get function from R script
    r_source = robjects.r['source']
    r_source(root + 'grf_experiments.R')
    r_get_te_predictions = robjects.globalenv['get_te_predictions']

    r_out = r_get_te_predictions(r_x, r_y, r_w, r_p, r_x_test, num_trees=num_trees)
    out = pandas2ri.ri2py_dataframe(r_out).values

    mses = [mean_squared_error(t_test, out[:, i]) for i in range(5)]

    return mses


def eval_setting_repeat_grf(n_train, num_trees=NUM_TREES_BASE, covariate_model=None,
                            repeats=N_REPEATS_BASE, n_test=N_TEST_BASE, d=1,
                            te_function=None, baseline_model=None, selection_bias=None,
                            propensity_model=None, error_model=None,
                            pre_dispatch='2*n_jobs', n_jobs=1, verbose=0,
                            root=PAPER_UTILS_ROOT):
    """
    Function repeatedly evaluates a setting
    """
    train = [i for i in range(n_train)]
    test = [i for i in range(n_train, n_train + n_test)]

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    # dispatch with different seeds
    scores = parallel(delayed(_eval_one_setting_grf)(train, test, n_train, n_test,
                                                     num_trees=num_trees,
                                                     d=d, selection_bias=selection_bias,
                                                     covariate_model=covariate_model,
                                                     te_function=te_function,
                                                     baseline_model=baseline_model,
                                                     propensity_model=propensity_model,
                                                     error_model=error_model,
                                                     seedy=i, root=root)
                      for i in range(repeats))

    zipped_scores = list(zip(*scores))
    plugscores = zipped_scores.pop(0)
    cfpscores = zipped_scores.pop(0)
    cfnpscores = zipped_scores.pop(0)
    ifpscores = zipped_scores.pop(0)
    ifnpscores = zipped_scores.pop(0)

    return plugscores, cfpscores, cfnpscores, ifpscores, ifnpscores


def eval_range_grf(range_dim, num_trees=NUM_TREES_BASE, dimension_range=True,
                   propensity_model=None,
                   repeats=N_REPEATS_BASE, covariate_model=None,
                   n_test=N_TEST_BASE, n_train=N_TRAIN_BASE, d=1,
                   te_function=None, baseline_model=None, error_model=None,
                   pre_dispatch='2*n_jobs', n_jobs=1, verbose=0):
    """
    Evaluate GRF perforamnce over a range of settings
    """
    resultframe = pd.DataFrame(columns=['plug_mean', 'cfp_mean', 'cfnp_mean', 'ifp_mean',
                                        'ifnp_mean',
                                        'plug_sd', 'cfp_sd', 'cfnp_sd', 'ifp_sd', 'ifnp_sd'])
    if dimension_range:
        for d in range_dim:
            print('Ambient dimension: {}'.format(d))
            scores = eval_setting_repeat_grf(n_train=n_train, num_trees=num_trees,
                                             repeats=repeats, n_test=n_test,
                                             d=d, te_function=te_function,
                                             baseline_model=baseline_model,
                                             propensity_model=propensity_model,
                                             error_model=error_model,
                                             covariate_model=covariate_model,
                                             pre_dispatch=pre_dispatch,
                                             n_jobs=n_jobs,
                                             verbose=verbose)
            # need possiblity for nans in case something goes wrong in small n regime
            means = [np.nanmean(np.array(x)) for x in scores]
            sds = [np.nanstd(np.array(x)) / np.sqrt(sum(~np.isnan(x))) for x in scores]
            resultframe.loc[d, :] = means + sds
    else:
        for n in range_dim:
            print('Number of training observations: {}'.format(n))
            scores = eval_setting_repeat_grf(n_train=n, num_trees=num_trees,
                                             repeats=repeats, n_test=n_test,
                                             d=d, te_function=te_function,
                                             baseline_model=baseline_model,
                                             covariate_model=covariate_model,
                                             propensity_model=propensity_model,
                                             error_model=error_model,
                                             pre_dispatch=pre_dispatch,
                                             n_jobs=n_jobs,
                                             verbose=verbose)
            # need possiblity for nans in case something goes wrong in small n regime
            means = [np.nanmean(np.array(x)) for x in scores]
            sds = [np.nanstd(np.array(x)) / np.sqrt(sum(~np.isnan(x))) for x in scores]
            resultframe.loc[d, :] = means + sds

    return resultframe
