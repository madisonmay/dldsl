from functools import partial, reduce
import copy
import operator

import numpy as np

from dldsl.expression import Expression
from dldsl.tensor import Tensor


class Atom(Expression):

    def __call__(self, *args, lhs, **kwargs):
        return self.items[0](*args, lhs=lhs, **kwargs)


class Negate(Expression):

    def __call__(self, *args, lhs, **kwargs):
        tensor = self.tensors[0]
        return Tensor.from_array(
            arr=-tensor.value, 
            axes=tensor.axes
        )


class Invert(Expression):

    def __call__(self, *args, lhs, **kwargs):
        tensor = self.tensors[0]
        return Tensor.from_array(
            arr=(1 / tensor.value), 
            axes=tensor.axes
        )


class Product(Expression):

    def __call__(self, *args, lhs, **kwargs):
        items = [item(*args, lhs=lhs, **kwargs) for item in self.items]
        ein_expr = ",".join(item._ein for item in items) + f"->{lhs._ein}"
        value = np.einsum(
            ein_expr,
            *(item.value for item in items)
        )
        return Tensor.from_array(
            value, 
            axes=lhs.axes
        )


class Sum(Expression):

    def __call__(self, *args, lhs, **kwargs):
        items = [item(*args, lhs=lhs, **kwargs) for item in self.items]

        bcast_axes = copy.deepcopy(lhs.axes)
        for item in items:
            for axis in item.axes:
                if axis not in bcast_axes:
                    bcast_axes.append(axis)
        
        items = [item._broadcast(axes=bcast_axes) for item in items]
        bcast_sum = reduce(operator.add, [item.value for item in items])
        bcast_tensor = Tensor.from_array(
            bcast_sum,
            axes=bcast_axes
        )
        return bcast_tensor