from dldsl.errors import AxisDisagreement


class Axis:

    def __init__(self, name):
        # Ignoring that axis can be an instance of int for now
        self.name = name
        self.value = None

    def concretize(self, value):
        if self.value is None:
            self.value = value
        elif self.value != value:
            raise AxisDisagreement(
                f"{self} was supplied contradictory concrete values of {self.value} and {value}"
            )
        
    def __repr__(self):
        return f"Axis({self.name})"
