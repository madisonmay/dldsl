from dldsl.tensor import Tensor


class Expression:
    
    def __init__(self, items, **kwargs):
        self.items = items
        self.tensors = []
        for item in items:
            if isinstance(item, Tensor):
                self.tensors.append(item)
            elif hasattr(item, "tensors"):
                self.tensors.extend(item.tensors)

    def __repr__(self):
        items_repr = ", ".join(repr(item) for item in self.items)
        return f"{self.__class__.__name__}([{items_repr}])"

    def __call__(self, *args, lhs, **kwargs):
        result = self.items[0](*args, lhs=lhs, **kwargs)
        return Tensor.from_array(
            result.value,
            result.axes
        )