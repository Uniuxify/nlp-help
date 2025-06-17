import inspect
from pandas.io import clipboard
import numpy as np


def func(name):
    clipboard.copy(dict_[name])

dict_ = {
    'test_func': '''
def test_func():
    print('success!')
'''
}
