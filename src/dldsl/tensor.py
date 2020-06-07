import uuid
import numpy as np

class Tensor:

    def __init__(self, name, index=None):
        self.name = name
        self.axes = [] if not index else index.axes
        self.value = None
        
    def concretize(self, tensor):
        self.value = tensor

    @property
    def shape(self):
        return self.value.shape

    @property
    def _ein(self):
        return "".join(axis._ein for axis in self.axes) 

    def _reduce(self, axes):
        # Reduce step
        # NOTE: currently unused
        reduction_dims = []
        remaining_axes = []
        for i, axis in enumerate(self.axes):
            if axis not in axes:
                reduction_dims.append(i)
        
        if reduction_dims:
            self.value = np.sum(self.value, axis=tuple(reduction_dims))
            self.axes = axes
        return self

    def _broadcast(self, axes):
        # Broadcast step
        missing_axes = set(axes) - set(self.axes)

        # Add placeholder dims for missing axes
        n_dims = len(self.axes)
        if missing_axes:
            self.value = np.expand_dims(
                self.value, list(range(n_dims, n_dims + len(missing_axes)))
            )
            self.axes += list(missing_axes)

        # Reorder axes to align with `axes` argument
        source_axes_idxs = list(range(len(self.axes)))
        axis_order = {axis.name: index for index, axis in enumerate(axes)}
        target_axes_idxs = []
        for axis in self.axes:
            target_axes_idxs.append(axis_order[axis.name])
        self.value = np.moveaxis(self.value, source_axes_idxs, target_axes_idxs)
        self.axes = axes
        
        return self

    def __repr__(self):
        return f'Tensor("{self.name}", {self.axes})'

    def __call__(self, *args, lhs, **kwargs):
        return self

    @classmethod
    def from_array(cls, arr, axes):
        tensor = cls(name=uuid.uuid4().hex)
        tensor.value = arr
        tensor.axes = axes
        assert len(tensor.axes) == len(tensor.value.shape)
        return tensor