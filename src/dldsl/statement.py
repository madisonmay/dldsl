import string

import numpy as np

from dldsl.errors import UnexpectedTensor, MissingTensor
from dldsl.operators import Product


class Statement:

    def __init__(self, return_tensor, expression):
        self._tensor_registry = {}
        self._axis_registry = {}
        self._ein_idx = 0
        self._ein_options = string.ascii_lowercase + string.ascii_uppercase
        self.return_tensor = return_tensor
        self.expression = expression

        for tensor in self.expression.tensors + [self.return_tensor]:
            for i, axis in enumerate(tensor.axes):
                # Within the context of a statement,
                # all axes with the same name should point to the same
                # axis object
                tensor.axes[i] = self._register_axis(axis)
            self._register_tensor(tensor)
        
    def _register_tensor(self, tensor):
        self._tensor_registry[tensor.name] = tensor
        return tensor

    def _register_ein(self, axis):
        character = self._ein_options[self._ein_idx]
        self._ein_idx += 1
        axis._register_ein(character)

    def _register_axis(self, axis):
        if axis.name not in self._axis_registry:
            self._axis_registry[axis.name] = axis
            self._register_ein(axis)

        return self._axis_registry[axis.name]

    def __call__(self, **kwargs):
        # We substitute our symbolic values with concrete values
        for name, tensor in kwargs.items():
            if name not in self._tensor_registry:
                raise UnexpectedTensor(name)
            
            symbolic_tensor = self._tensor_registry[name]
            assert len(tensor.shape) == len(symbolic_tensor.axes)

            for axis, concrete_value in zip(symbolic_tensor.axes, tensor.shape):
                axis.concretize(concrete_value)

            symbolic_tensor.concretize(tensor)

        for symbolic_tensor in self.expression.tensors:
            if symbolic_tensor.value is None:
                raise MissingTensor(symbolic_tensor.name)
        
        self.return_tensor.concretize(np.zeros([axis.value for axis in self.return_tensor.axes]))

        tensor = self.expression(lhs=self.return_tensor)

        # One final einsum to deal with base case of reductions on a singular tensor
        # Possibly can handle this as part of the parser grammar?
        return Product([tensor])(lhs=self.return_tensor).value
    
    def __repr__(self):
        return f"Statement(return_tensor={self.return_tensor}, expression={self.expression})"
