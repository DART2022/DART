# coding: UTF-8
"""
    @author: samuel ko
    @date:   2019.12.13
    @readme: Miscellaneous utility classes and functions.
"""

import re
import importlib
import torch
import os
import sys
import types
from typing import Any, List, Tuple, Union


def SaveModel(encoder, decoder, dir, epoch):
    params = {}
    params['encoder'] = encoder.state_dict()
    params['decoder'] = decoder.state_dict()
    torch.save(params, os.path.join(dir, '_params_{}.pt'.format(epoch)))
