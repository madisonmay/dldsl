from dldsl.tensor import Tensor


class Expression:
    
    def __init__(self, items):
        self.items = items
        self.tensors = [item for item in self.items if isinstance(item, Tensor)]

    def __repr__(self):
        items_repr = ", ".join(repr(item) for item in self.items)
        return f"Expression([{items_repr}])"
