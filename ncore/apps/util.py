"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc, Sonali Parbhoo, Harvard University
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function

import sys
import time
import numpy as np
from datetime import datetime


def clip_percentage(value):
    return max(0., min(1., float(value)))


def isfloat(value):
    if value is None:
        return False

    try:
        float(value)
        return True
    except ValueError:
        return False


def random_cycle_generator(num_origins, seed=505):
    while 1:
        random_generator = np.random.RandomState(seed)
        samples = random_generator.permutation(num_origins)
        for sample in samples:
            yield sample


def resample_with_replacement_generator(array):
    while 1:
        for _ in range(len(array)):
            sample_idx = np.random.randint(0, len(array))
            yield array[sample_idx]


def get_gpu_devices():
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if 'GPU' in x.device_type]


def get_cpu_devices():
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if 'CPU' in x.device_type]


def get_num_available_gpus():
    return len(get_gpu_devices())


def error(*msg, **kwargs):
    log(*msg, log_level="ERROR", **kwargs)


def warn(*msg, **kwargs):
    log(*msg, log_level="WARN", **kwargs)


def info(*msg, **kwargs):
    log(*msg, log_level="INFO", **kwargs)


def log(*msg, **kwargs):
    sep = kwargs.pop('sep', " ")
    end = kwargs.pop('end', "\n")
    log_level = kwargs.pop('log_level', "INFO")
    with_timestamp = kwargs.pop('with_timestamp', True)

    initial_sep = " " if sep == "" else ""
    timestamp = " [{:.7f}]".format(time.time()) if with_timestamp else ""

    print(log_level + timestamp + ":" + initial_sep, *msg, sep=sep, end=end,
          file=sys.stdout if log_level == "INFO" else sys.stderr)


def report_duration(task, duration):
    log(task, "took", duration, "seconds.")


def time_function(task_name):
    def time_function(func):
        def func_wrapper(*args, **kargs):
            t_start = time.time()
            return_value = func(*args, **kargs)
            t_dur = time.time() - t_start
            report_duration(task_name, t_dur)
            return return_value
        return func_wrapper
    return time_function


def get_date_format_string():
    return '%Y-%m-%d %H:%M:%S'


def convert_str_to_date(date_str):
    date = datetime.strptime(date_str, get_date_format_string())
    return date


def convert_date_to_str(date):
    date_str = date.strftime(get_date_format_string())
    return date_str


def convert_int_to_indicator_list(x, min_length=1):
    if not isinstance(x, int) and not isinstance(x, np.int64):
        raise AssertionError("__x__ must be an integer value.")

    if x == 0:
        indicator_list = [0]
    else:
        indicator_list = []
        while x:
            indicator_list.append(x % 2)
            x >>= 1

        indicator_list = list(reversed(indicator_list))

    if len(indicator_list) < min_length:
        indicator_list = [0]*(min_length - len(indicator_list)) + indicator_list
    return indicator_list


def convert_indicator_list_to_int(indicator_list):
    indicator_list = indicator_list.astype(int)
    int_value = 0
    for indicator in indicator_list:
        int_value = (int_value << 1) | indicator
    return int_value


def mixed_distance(x1, x2, discrete_indices):
    assert len(x1) == len(x2)
    continuous_indices = np.array(list(set(range(len(x1))) - set(discrete_indices)), dtype=int)
    discrete_dist = jaccard_distance(x1[discrete_indices], x2[discrete_indices])
    continuous_dist = np.sum(np.absolute(x1[continuous_indices] - x2[continuous_indices]))
    num_indices = float(len(x1))
    return len(discrete_indices)/num_indices*discrete_dist + len(continuous_indices)/num_indices*continuous_dist


def jaccard_distance(x1, x2):
    intersection = sum(map(lambda xx1, xx2: int(xx1) & int(xx2), x1, x2))
    union = sum(x1) + sum(x2) - intersection
    distance = 1 - intersection / float(union)
    return distance
