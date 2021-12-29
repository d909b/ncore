"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc

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
import six
import h5py
import numpy as np
from abc import ABCMeta
from collections import defaultdict
from collections import OrderedDict
from typing import Union, List, Any, IO, AnyStr, NoReturn, Dict


@six.add_metaclass(ABCMeta)
class BaseDataSource(object):
    COVARIATES_KEY = "covariates"
    SHAPES_KEY = "shapes"
    ROWNAMES_KEY = "rownames"
    COLNAMES_KEY = "colnames"
    COLUMN_VALUE_MAP_KEYS_KEY = "colmapkeys_col{:d}"
    COLUMN_VALUE_MAP_VALUES_KEY = "colmapvalues_col{:d}"
    DATASET_NAME = "name"
    DATASET_VERSION = "version"

    def __init__(self, hd5_file: Union[AnyStr, IO], included_indices: List[int] = None):
        self.hd5_file = hd5_file

        if included_indices is None:
            # Include all indices by default.
            self.included_indices = np.array(list(range(self._underlying_length())))
        else:
            self.included_indices = np.array(included_indices)

        self._build_row_index()
        self._check_hd5_file_format()
        self._check_included_ids_format()

    def get_shape(self):
        with h5py.File(self.hd5_file, "r") as hd5_file:
            shape = hd5_file[BaseDataSource.COVARIATES_KEY].shape

            has_dynamic_dimensions = len(shape) == 1
            if has_dynamic_dimensions:
                if BaseDataSource.SHAPES_KEY in hd5_file:
                    original_shape = (None,)*len(hd5_file[BaseDataSource.SHAPES_KEY].shape)
                else:
                    original_shape = (None,)
                shape = shape + original_shape
                if shape[-1] is None:
                    shape = shape + (1,)
            return shape

    def _build_row_index(self):
        row_names = self.get_row_names()
        row_index = defaultdict(list)
        for name, index in zip(row_names, range(len(row_names))):
            row_index[name].append(index)

        has_multiple = max(map(len, row_index.values())) > 1
        if not has_multiple:
            # Flatten index if there is a 1-1 mapping.
            self.row_index = OrderedDict(zip(row_names, range(len(row_names))))
        else:
            self.row_index = row_index

    def _check_hd5_file_format(self) -> NoReturn:
        with h5py.File(self.hd5_file, "r") as hd5_file:
            assert BaseDataSource.COVARIATES_KEY in hd5_file
            assert BaseDataSource.ROWNAMES_KEY in hd5_file
            assert BaseDataSource.COLNAMES_KEY in hd5_file
            if len(hd5_file[BaseDataSource.COVARIATES_KEY].shape) == 2:
                assert len(hd5_file[BaseDataSource.COLNAMES_KEY]) == hd5_file[BaseDataSource.COVARIATES_KEY].shape[-1]
            elif len(hd5_file[BaseDataSource.COVARIATES_KEY].shape) == 1:
                assert len(hd5_file[BaseDataSource.COLNAMES_KEY]) == 1
            assert len(hd5_file[BaseDataSource.ROWNAMES_KEY]) == hd5_file[BaseDataSource.COVARIATES_KEY].shape[0]
            assert BaseDataSource.DATASET_NAME in hd5_file.attrs
            assert BaseDataSource.DATASET_VERSION in hd5_file.attrs

    def _check_included_ids_format(self) -> NoReturn:
        all_identifiers = set(range(self._underlying_length()))
        assert set(self.included_indices).issubset(all_identifiers), \
            "All included IDs must be present in the identifier list."

    def _underlying_length(self):
        return self.get_shape()[0]

    def __len__(self) -> int:
        return len(self.included_indices)

    def __getitem__(self, index: int) -> Any:
        forwarded_index = self.included_indices[index]
        with h5py.File(self.hd5_file, "r") as hd5_file:
            x = hd5_file[BaseDataSource.COVARIATES_KEY][forwarded_index]
            if BaseDataSource.SHAPES_KEY in hd5_file:
                original_shape = hd5_file[BaseDataSource.SHAPES_KEY][forwarded_index]
                x = x.reshape(original_shape)
            return x

    def get_name(self) -> AnyStr:
        with h5py.File(self.hd5_file, "r") as hd5_file:
            return hd5_file.attrs[BaseDataSource.DATASET_NAME]

    def get_version(self) -> AnyStr:
        with h5py.File(self.hd5_file, "r") as hd5_file:
            return hd5_file.attrs[BaseDataSource.DATASET_VERSION]

    def get_data(self):
        with h5py.File(self.hd5_file, "r") as hd5_file:
            x = hd5_file[BaseDataSource.COVARIATES_KEY]
            return x[()]

    def get_column_value_maps(self) -> List[Dict[int, AnyStr]]:
        column_value_maps = []
        with h5py.File(self.hd5_file, "r") as hd5_file:
            column_names = self.get_column_names()
            for idx, column in enumerate(column_names):
                keys_name = BaseDataSource.COLUMN_VALUE_MAP_KEYS_KEY.format(idx)
                values_name = BaseDataSource.COLUMN_VALUE_MAP_VALUES_KEY.format(idx)
                if keys_name in hd5_file and values_name in hd5_file:
                    keys, values = hd5_file[keys_name], hd5_file[values_name]
                    if len(keys) != len(values):
                        raise AssertionError(
                            "Malformed hd5 data source. Value map keys and values must be of the same length."
                        )
                    column_value_map = dict(zip(keys, values))
                else:
                    column_value_map = None
                column_value_maps.append(column_value_map)
        return column_value_maps

    def inverse_transform(self, item):
        column_value_maps = self.get_column_value_maps()
        if len(item) != len(column_value_maps):
            raise AssertionError(f"Item must be the same length as the number of columns in the data source."
                                 f"Expected {len(column_value_maps):d} but was {len(item):d}.")

        item = list(map(lambda value, value_map:
                        value_map[value] if value_map is not None else value,
                        item, column_value_maps))
        return item

    def is_variable_length(self) -> bool:
        if len(self.row_index) == 0:
            return False
        key = self.row_index.items()[0]
        is_variable_length = isinstance(self.row_index[key], list)
        return is_variable_length

    def get_column_names(self) -> List[AnyStr]:
        with h5py.File(self.hd5_file, "r") as hd5_file:
            column_names = hd5_file[BaseDataSource.COLNAMES_KEY]
            return column_names[()].tolist()

    def get_column_name(self, index) -> AnyStr:
        with h5py.File(self.hd5_file, "r") as hd5_file:
            column_name = hd5_file[BaseDataSource.COLNAMES_KEY][index]
            return column_name

    def get_row_names(self) -> List[AnyStr]:
        with h5py.File(self.hd5_file, "r") as hd5_file:
            row_names = hd5_file[BaseDataSource.ROWNAMES_KEY]
            row_names = [row_names[index] for index in self.included_indices]
            return row_names

    def get_row_name(self, index) -> AnyStr:
        forwarded_index = self.included_indices[index]
        with h5py.File(self.hd5_file, "r") as hd5_file:
            row_name = hd5_file[BaseDataSource.ROWNAMES_KEY][forwarded_index]
            return row_name

    def get_by_row_name(self, row_name) -> Any:
        idx = self.row_index[row_name]
        if isinstance(idx, list):
            return [self[i] for i in idx]
        else:
            return self[idx]
