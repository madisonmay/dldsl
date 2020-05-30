# Not that kind of transformer
from functools import partial

from lark import Transformer, Tree

from dldsl.axis import Axis
from dldsl.index import Index
from dldsl.tensor import Tensor
from dldsl.operator import Operator
from dldsl.expression import Expression
from dldsl.statement import Statement


class DLTransformer(Transformer):
    
    def __init__(self, *args, **kwargs):
        self.axis_registry = {}
    
    def statement(self, items):
        return_tensor, _, expression = items 
        return Statement(return_tensor=return_tensor, expression=expression)

    def operator(self, items):
        return Operator(items[0])

    def expr(self, items):
        return Expression(items)

    def tensor(self, items):
        name = items[0]
        index = None if len(items) is 1 else items[1]
        tensor = Tensor(name=name, index=index)
        return tensor
    
    def index(self, items):
        return Index(items)

    def axis(self, items):
        return Axis(items[0].value)


def transform(tree: Tree):
    # Transformer will likely be stateful
    transformer = DLTransformer()
    return transformer.transform(tree)
