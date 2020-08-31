"""
Author: Alicia Curth
Module implements Group-IF-learner from Curth, Alaa and van der Schaar (2020), for treatment
effects with known propensity scores
"""
import numpy as np
import pandas as pd
import statsmodels as sm

from .base import CATE_NAME, ht_te_transformation
from .base_learners import IFLearnerTE, PlugInTELearner


class GroupIFLearner(IFLearnerTE):
    """
    Class implementing the Group-IF-learner from Curth, Alaa and van der Schaar (2020) for
    treatment effect estimation.

    Parameters
    ----------
    te_estimator: estimator
        estimator for second stage
    base_estimator: estimator, default None
        estimator for first stage. Can be None, then te_estimator is used
    setting: str or callable, default 'CATE'
        The treatment effect setting to be considered. Currently built-in support for 'CATE' or
        'RR'. Can also pass a callable that is a plug-in function of the two potential outcome
        regressions.
    binary_y: bool, default False
        Whether the outcome data is binary
    plug_in_corrected: bool, default True
        Whether to use plug-in bias-corrected first stage model for grouping (IF-learner)
    baseline_adjustment: bool, default True
        Whether to use baseline adjustment in Second stage (for improved precision)
    efficient_est: bool, default True
        Whether to use efficient second stage estimator. If False and setting 'CATE', will use
        Chernozhukov et al. (2018)'s estimator instead
    honest: bool, default True
        Whether to perform honest estimation by splitting data in auxiliary and estimation sample
        (recommended)
    n_folds: int, default 10
        Number of cross-fitting folds
    random_state: int, default 42
        random state to use for cross-fitting splits
    n_groups: int, default 5
        number of groups to be created on basis of quantiles of treatment effects
    """
    def __init__(self, te_estimator, base_estimator=None,
                 setting=CATE_NAME, binary_y: bool = False,
                 plug_in_corrected: bool = True, baseline_adjustment: bool = True,
                 efficient_est: bool = True, honest: bool = False,
                 n_folds=10, random_state=42, n_groups: int = 5, ):
        super().__init__(te_estimator=te_estimator, fit_base_model=baseline_adjustment,
                         base_estimator=base_estimator, fit_propensity_model=False,
                         n_folds=n_folds, setting=setting, binary_y=binary_y,
                         random_state=random_state)
        self.n_groups = n_groups
        self.plug_in_corrected = plug_in_corrected
        self.baseline_adjustment = baseline_adjustment
        self.efficient_est = efficient_est
        self.honest = honest

    def _prepare_self(self):
        # clean up
        super()._prepare_self()

        # make a learner IFLearner within this learner
        if self.plug_in_corrected:
            self._plug_in_model = IFLearnerTE(te_estimator=self.te_estimator, fit_base_model=True,
                                              base_estimator=self.base_estimator,
                                              setting=self.setting, binary_y=self.binary_y,
                                              propensity_estimator=self.propensity_estimator,
                                              double_sample_split=self.double_sample_split,
                                              n_folds=self.n_folds,
                                              random_state=self.random_state)
        else:
            self._plug_in_model = PlugInTELearner(base_estimator=self.base_estimator,
                                                  setting=self.setting)

    def _fit(self, X, y, w, p=None):
        n = len(y)
        if self.honest:
            # create heldout data set for estimation
            idx_part = np.random.choice(range(n), size=int(np.round(0.5 * n)),
                                        replace=False)
            mask_part = np.zeros(n, dtype=bool)
            mask_part[idx_part] = True

            # split sample
            X_est, y_est, w_est, p_est = X[~mask_part, :], y[~mask_part], w[~mask_part], p[
                ~mask_part]
            X, y, w, p = X[mask_part, :], y[mask_part], w[mask_part], p[mask_part]
            n = X.shape[0]

        # plug-in step
        self._plug_in_model.fit(X, y, w, p)

        # given thresholds compute taus -- either honest or not
        if self.honest:
            # replace estimation data
            X, y, w, p = X_est, y_est, w_est, p_est

        self._calc_taus(X, y, w, p)

    def _get_te_and_set_thresholds(self, X):

        # determine groups on basis of treatment effects
        te = self._plug_in_model.predict(X)

        # determine quantiles
        thresholds = np.quantile(te,
                                 q=np.linspace(0, 1, self.n_groups + 1))[1:self.n_groups]

        # in case there are constants
        self._thresholds = np.unique(thresholds)

        if self._thresholds[-1] == max(te):
            # for the case that there will be an empty final group
            self._thresholds = self._thresholds[:-1]

        # count groups
        self.n_groups = len(self._thresholds) + 1

        return te

    def _calc_taus(self, X, y, w, p):

        te = self._get_te_and_set_thresholds(X)

        # calculate the TEs by group
        # initialise variables
        taus = np.zeros(self.n_groups)
        tau_vars = np.zeros(self.n_groups)
        grouped = np.zeros(len(y), dtype=bool)
        n_g = np.zeros(self.n_groups)
        n_t = np.zeros(self.n_groups)

        if not self.efficient_est:
            group_indicators = np.zeros((len(y), self.n_groups), dtype=int)

        if self.baseline_adjustment:
            # get regression adjustment terms
            _, mu_0, mu_1 = self._plug_in_model.predict(X, return_po=True)

        # compute taus for first k-1 groups
        for i in range(self.n_groups - 1):
            member = (te <= self._thresholds[i]) & (~grouped)

            if np.sum(1 - w[member]) == 0 or np.sum(w[member]) == 0:
                raise ValueError(
                    'Only one of treatment and control group is present in group {} '
                    'cannot compute the group effect'.format(i + 1))

            if self.baseline_adjustment:
                if self.efficient_est:
                    taus[i] = self._group_tau(y[member], w[member], p[member], mu_0[member],
                                              mu_1[member])
                    tau_vars[i] = self._group_var(y[member], w[member], p[member], mu_0[member],
                                                  mu_1[member])
                else:
                    # make group indicators for GATES
                    group_indicators[member, i] = 1

            else:
                taus[i] = self._group_tau(y[member], w[member], p[member])
                tau_vars[i] = self._group_var(y[member], w[member], p[member])

            grouped[member] = True
            n_g[i] = np.sum(member)
            n_t[i] = np.sum(w[member])

        # make last group with leftovers
        if np.sum(1 - w[~grouped]) == 0 or np.sum(w[~grouped]) == 0:
            raise ValueError(
                'Only one of treatment and control group is present in group {} '
                'cannot compute the group effect'.format(self.n_groups))

        n_g[self.n_groups - 1] = np.sum(~grouped)
        n_t[self.n_groups - 1] = np.sum(w[~grouped])

        if self.baseline_adjustment:
            if self.efficient_est:
                taus[self.n_groups - 1] = self._group_tau(y[~grouped], w[~grouped],
                                                          p[~grouped], mu_0[~grouped],
                                                          mu_1[~grouped])
                tau_vars[self.n_groups - 1] = self._group_var(y[~grouped], w[~grouped],
                                                              p[~grouped], mu_0[~grouped],
                                                              mu_1[~grouped])
            else:
                # make group indicators for GATES
                group_indicators[~grouped, self.n_groups - 1] = 1
                # make GATES treatment effects
                self._calc_taus_GATES(y, w, p, group_indicators, mu_0)

        else:
            taus[self.n_groups - 1] = self._group_tau(y[~grouped], w[~grouped],
                                                      p[~grouped], None, None)
            tau_vars[self.n_groups - 1] = self._group_var(y[~grouped], w[~grouped],
                                                          p[~grouped], None, None)

        if self.efficient_est or not self.baseline_adjustment:
            # then treatment effects are not yet saved
            if np.isnan(taus).any():
                raise ValueError('One or more of the within-group treatment effects could not be '
                                 'computed.')
            self._taus = taus
            self._tau_var = tau_vars

        self._n_g = n_g
        self._n_t = n_t

    def _calc_taus_GATES(self, y, w, p, group_indicators, mu_0):
        # do regression adjustment as suggested in Chernozhukov et al (2018)
        orthogonal_treat = w - p
        orthogonal_group = group_indicators * orthogonal_treat[:, np.newaxis]
        weights = 1 / (p * (1 - p))

        X = pd.DataFrame(np.c_[orthogonal_group, mu_0])
        wls_model = sm.WLS(y, X, weights=weights)
        res = wls_model.fit()
        self._taus = res.params[:self.n_groups]

    def predict(self, X, return_po=False):
        """
        Get group-wise treatment effects for new set of data

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError('Group-IF-Learners have no po-model')

        # predict treatment effects
        n = X.shape[0]
        tau_tilde = self._plug_in_model.predict(X)

        te_est = np.zeros(n)
        grouped = np.zeros(n, dtype=bool)

        # get predictions
        for i in range(self.n_groups - 1):
            member = (tau_tilde < self._thresholds[i]) & (~grouped)
            te_est[member] = self._taus[i]
            grouped[member] = True

        # assign last people to highest group
        te_est[~grouped] = self._taus[self.n_groups - 1]

        return te_est

    def _group_tau(self, y, w, p=None, mu_0=None, mu_1=None):
        if np.sum(1 - w) == 0 or np.sum(w) == 0:
            raise ValueError('Only one group is present, cannot compute group-TE')
        if self.baseline_adjustment:
            # influence function estimator
            return np.average(self._eif(y=y, w=w, p=p, mu_0=mu_0, mu_1=mu_1))
        else:
            # IPW/HT estimate
            if self.setting == 'RR':
                return rr_from_means(y, w, p)
            else:
                return np.average(ht_te_transformation(y, w, p))

    def _group_var(self, y, w, p=None, mu_0=None, mu_1=None):
        if np.sum(1 - w) == 0 or np.sum(w) == 0:
            raise ValueError('Only one group is present, cannot compute variance of group-TE')
        n = len(w)
        if self.baseline_adjustment:
            return 1 / n * np.var(self._eif(y=y, w=w, p=p, mu_0=mu_0, mu_1=mu_1))
        else:
            if self.setting == 'RR':
                return np.nan
            return 1 / n * np.var(ht_te_transformation(y, w, p))


def rr_from_means(y, w, p):
    """
    Compute risk ratios from Horvitz-Thompson weighted means of outcomes.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        The treatment propensity, estimated or known. Can be None, then p=0.5 is assumed

    Returns
    -------
    rr_mean: float
        Risk ratio of HT-means
    """
    if p is None:
        # assume equal
        p = np.full(len(y), 0.5)

    if np.sum(1 - w) == 0 or np.sum(w) == 0:
        raise ValueError('Only one group is present, cannot compute RR.')
    if np.mean((1 - w) / (1 - p) * y) == 0:
        raise ValueError('cannot compute ratio because denominator is zero')
    return np.mean((w / p) * y) / np.mean((1 - w) / (1 - p) * y)
