from dldsl.errors import AxisDisagreement


class Axis:

    def __init__(self, name):
        self.name = name
        self.value = None
        self._ein = None
    
    def _register_ein(self, character):
        self._ein = character

    def concretize(self, value):
        if self.value is None:
            self.value = value
        elif self.value != value:
            raise AxisDisagreement(
                f"{self} was supplied contradictory concrete values of {self.value} and {value}"
            )
        
    def __repr__(self):
        return f"Axis({self.name})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name
