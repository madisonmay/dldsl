from dldsl.parser import parse
from dldsl.transformer import transform


def op(s):
    """
    Parse string and transform into abstract statement
    """
    parsed = parse(s)
    return transform(parsed)
