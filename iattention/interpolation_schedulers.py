# -*- coding: utf-8 -*-
import numpy as np


def get_linear_value(current_index, start_value, total_steps, end_value=0, **kwargs):
    values = np.linspace(start_value, end_value, total_steps, dtype=np.float32)
    values = values / start_value * start_value
    return values[current_index]


def get_cosine_value(current_index, start_value, total_steps, end_value=0, **kwargs):
    values = np.linspace(end_value, total_steps, total_steps, dtype=np.float32)
    values = np.cos(values * np.pi / total_steps)
    values = (values + 1) * start_value / 2
    return values[current_index]


def get_ema_value(current_index, start_value, eta, **kwargs):
    value = start_value * eta ** current_index
    return value


INTERPOLATION_SCHEDULERS = {
    'ema': get_ema_value,
    'cos': get_cosine_value,
    'linear': get_linear_value,
}
