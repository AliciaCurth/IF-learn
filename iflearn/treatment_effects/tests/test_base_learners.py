"""
Author: Alicia Curth
Unittests for the base_learners module
"""
import pytest
from iflearn.treatment_effects.base import _get_po_function, _get_te_eif
from iflearn.treatment_effects.base_learners import BaseTEModel, PlugInTELearner, IFLearnerTE
from iflearn.simulation_utils.treatment_effects import make_te_data

import pandas as pd
import numpy as np


def test_basetemodel_constructor():
    # test that the right errors are thrown
    pass

def test_iflearnerte_constructor():
    # test that the right errors are thrown
    pass

def test_settingscheck_eif():
    # test that the setting checker works
    notcallable = pd.DataFrame(np.array([0,1]))
    with pytest.raises(ValueError):
        _get_te_eif('nonsense')

    with pytest.raises(ValueError):
        _get_te_eif(notcallable)

    with pytest.raises(ValueError):
        _get_po_function('nonsense')

    with pytest.raises(ValueError):
        _get_po_function(notcallable)
