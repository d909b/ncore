"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc, Sonali Parbhoo, Harvard University
Copyright (C) 2020  Patrick Schwab, F. Hoffmann-La Roche Ltd
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

import os
import six
import numpy as np
from os.path import join
from datetime import datetime
from ncore.apps.util import info, convert_date_to_str


def to_r_object(obj):
    if isinstance(obj, tuple):
        ret_val = "list("
        ret_val += "'" + convert_date_to_str(obj[0]) + "',"
        ret_val += str(obj[1])
        ret_val += ")"
        return ret_val
    elif isinstance(obj, six.string_types):
        return "\"" + str(obj) + "\""
    elif isinstance(obj, dict):
        return to_r_hash(obj, to_file=False)
    else:
        return str(obj)


def to_r_array(array, idx=None, output_dir=None, name_format=None, to_file=True, as_list=False):
    if as_list:
        list_type = "list"
    else:
        list_type = "c"

    r_array = ""
    r_array += "{}(".format(list_type) + ",".join(map(to_r_object, array)) + ")"

    if to_file:
        tmp_file_path = os.path.join(output_dir, name_format.format(idx))
        with open(tmp_file_path, "w") as tmp_file:
            tmp_file.write(r_array)
        r_array = "\"" + tmp_file_path + "\""
    return r_array


def to_r_hash(dictionary, idx=None, output_dir=None, name_format=None, to_file=True):
    all_vars = []
    r_hash = "hash("
    for k in sorted(dictionary.keys()):
        v = dictionary[k]
        if isinstance(v, dict):
            r_value = to_r_hash(v, to_file=False)
        else:
            r_value = to_r_array(v, to_file=False, as_list=True)
        all_vars.append("\"" + k + "\"=" + r_value)
    r_hash += ",".join(all_vars) + ")"
    if to_file:
        tmp_file_path = os.path.join(output_dir, name_format.format(idx))
        with open(tmp_file_path, "w") as tmp_file:
            tmp_file.write(r_hash)
        r_hash = "\"" + tmp_file_path + "\""
    return r_hash


def get_r_script(file_name):
    this_directory = os.path.dirname(os.path.realpath(__file__))
    timeline_r_script = join(this_directory, file_name)
    return timeline_r_script


def invoke_r_script(script_name, args, output_dir, file_name, with_print, name_format=None):
    script_path = get_r_script(script_name)

    if name_format is None:
        name_format = "{prefix}".format(prefix=file_name) + "_tmp_{}.txt"

    arg_dir = os.path.join(output_dir, "reproducibility")
    if not os.path.isdir(arg_dir):
        os.mkdir(arg_dir)

    converted_args = list(map(lambda idx, arg: to_r_array(arg, idx, arg_dir, name_format)
                                               if isinstance(arg, np.ndarray) else
                                               "\"" + arg + "\"" if isinstance(arg, six.string_types) else
                                               to_r_hash(arg, idx, arg_dir, name_format) if isinstance(arg, dict) else
                                               "\"" + convert_date_to_str(arg) + "\"" if isinstance(arg, datetime) else
                                               str(arg), range(len(args)), args))
    command_string = (
        "Rscript " + script_path + " " +
        " ".join(converted_args)
    )

    param_string_file = join(output_dir, file_name + ".txt")
    with open(param_string_file, "w") as fp:
        fp.write(command_string)
    try:
        os.popen(command_string).read()
    except OSError as err:
        info("INFO: OSError:" + str(err))
    if with_print:
        info("INFO: Saved plot to", join(output_dir, file_name))
