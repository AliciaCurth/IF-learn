"""
Author: Alicia Curth
Module contains important transformations, such as EIFs, Horvitz Thompson transformations,
functions of potential outcomes
"""
import warnings
import numpy as np
# define constants
CATE_NAME = 'CATE'
RR_NAME = 'RR'


# EIFs --------------------------------------------------------------------------
def eif_transformation_CATE(y, w, p, mu_0, mu_1):
    w_1 = w / p
    w_0 = (1 - w) / (1 - p)
    return (w_1 - w_0) * y + ((1 - w_1) * mu_1 - (1 - w_0) * mu_0)


def eif_transformation_RR(y, w, p, mu_0, mu_1):
    if np.sum(mu_0 == 0) > 0:
        raise ValueError('cannot compute RR EIF transformation if mu_0 is zero')
    w_1 = w / p
    w_0 = (1 - w) / (1 - p)
    ic_1 = 1/mu_0 * (w_1*y + (1-w_1)*mu_1 - mu_1)
    ic_0 = - mu_1/(mu_0**2)*(w_0*y + (1-w_0)*mu_0 - mu_0)
    return 1*(ic_0 + ic_1) + mu_1/mu_0


# PO functions --------------------------------------------------------------
def po_function_CATE(mu_0, mu_1):
    return mu_1 - mu_0


def po_function_RR(mu_0, mu_1):
    return mu_1/mu_0


# Horvitz Thompson transformations -----------------------------------------------
def ht_te_transformation(y, w, p=None):
    # horvitz-thompson transformation for treatment effects
    if p is None:
        # assume equal propensities
        p = np.full(len(y), 0.5)
    return (w / p - (1 - w) / (1 - p)) * y


# Helpers and constants -----------------------------------------------------------
TE_SETTINGS = [CATE_NAME, RR_NAME]
TE_SETTINGS_EIF = {CATE_NAME: eif_transformation_CATE, RR_NAME: eif_transformation_RR}
TE_SETTINGS_PO_FUNCTIONS = {CATE_NAME: po_function_CATE, RR_NAME: po_function_RR}
TE_SETTINGS_BINARY = [RR_NAME]


def _get_te_eif(setting, binary=None):
    # set EIF
    if isinstance(setting, str):
        try:
            _eif = TE_SETTINGS_EIF[setting]
        except KeyError:
            raise ValueError('%r is not a valid TE setting value. '
                             'Use iflearn.treatment_effects.TE_SETTINGS '
                             'to get valid options.' % setting)
        if binary is None:
            return _eif

        if not binary and setting in TE_SETTINGS_BINARY:
            warnings.warn("You chose to work with a setting made for binary_y data but set "
                          "binary_y=False.")
    elif callable(setting):
        _eif = setting
    else:
        raise ValueError('Setting should be either a string in '
                         'iflearn.treatment_effects.TE_SETTINGS or a callable eif.')

    return _eif


def _get_po_function(setting, binary=None):
    if isinstance(setting, str):
        try:
            _po_function = TE_SETTINGS_PO_FUNCTIONS[setting]
        except KeyError:
            raise ValueError('%r is not a valid TE setting value. '
                             'Use iflearn.treatment_effects.TE_SETTINGS '
                             'to get valid options.' % setting)
        if binary is None:
            return _po_function
        if not binary and setting in TE_SETTINGS_BINARY:
            warnings.warn("You chose to work with a setting made for binary_y data but set "
                          "binary_y=False.")
    elif callable(setting):
        _po_function = setting
    else:
        raise ValueError('Setting should be either a string in '
                         'iflearn.treatment_effects.TE_SETTINGS or a callable po_function.')
    return _po_function
