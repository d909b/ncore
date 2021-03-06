"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
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
import six
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class FeatureType(object):
    @abstractmethod
    def is_discrete(self):
        raise NotImplementedError()


class FeatureTypeUnknown(FeatureType):
    def is_discrete(self):
        return False


class FeatureTypeContinuous(FeatureType):
    def is_discrete(self):
        return False


class FeatureTypeDiscrete(FeatureType):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def is_discrete(self):
        return True


class FeatureTypeMissingIndicator(FeatureTypeDiscrete):
    def __init__(self):
        super(FeatureTypeMissingIndicator, self).__init__(num_classes=2)
