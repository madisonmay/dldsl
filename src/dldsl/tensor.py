class Tensor:

    def __init__(self, name, index=None):
        self.name = name
        self.axes = [] if not index else index.axes
        self.value = None
        
    def concretize(self, tensor):
        self.value = tensor

    def __repr__(self):
        return f'Tensor("{self.name}", {self.axes})'
