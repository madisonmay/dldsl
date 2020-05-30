import numpy as np

from dldsl.errors import UnexpectedTensor, MissingTensor


class Statement:

    def __init__(self, return_tensor, expression):
        self._tensor_registry = {}
        self._axis_registry = {}
        self.return_tensor = return_tensor
        self.expression = expression
        for tensor in self.expression.tensors + [self.return_tensor]:
            for i, axis in enumerate(tensor.axes):
                # Within the context of a statement,
                # all axes with the same name should point to the same
                # axis object
                tensor.axes[i] = self.register_axis(axis)
            self.register_tensor(tensor)
        
    def register_tensor(self, tensor):
        self._tensor_registry[tensor.name] = tensor
        return tensor

    def register_axis(self, axis):
        if axis.name not in self._axis_registry:
            self._axis_registry[axis.name] = axis
        return self._axis_registry[axis.name]

    def __call__(self, **kwargs):
        # Where the magic happens
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

    def __repr__(self):
        return f"Statement(return_value={self.return_value}, expression={self.expression})"
