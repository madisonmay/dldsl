class Index:

    def __init__(self, axes):
        self.axes = axes

    def __repr__(self):
        axes_repr = ", ".join(str(axis) for axis in self.axes)
        return f"[{axes_repr}]"
