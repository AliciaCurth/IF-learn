"""
Author: Alicia Curth
Module implements general utils
"""
import copy
from pygam.terms import TermList


def clone(estimator, safe=True):
    """Constructs a new te_estimator with the same parameters.

    Adapted from sklearn.clone -- adaptation necessary due to incompatibility
    of pygam with sklearn framework

    Clone does a deep copy of the model in an te_estimator
    without actually copying attached data. It yields a new te_estimator
    with the same parameters that has not been fit on any data.

    Parameters
    ----------
    estimator : te_estimator object, or list, tuple or set of objects
        The te_estimator or group of estimators to be cloned

    safe : boolean, optional
        If safe is false, clone will fall back to a deep copy on objects
        that are not estimators.

    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params') or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        # adapted here to allow term list
        if isinstance(param, TermList):
            new_object_params[name] = TermList(param)
        else:
            new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        # adapted here to allow mutable defaults (e.g. list)
        if param1 is not param2 and param1 != param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'either does not set or modifies parameter %s' %
                               (estimator, name))
    return new_object
