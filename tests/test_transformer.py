from dldsl import parse, transform
from dldsl.statement import Statement


def test_matmul_transform():
    op = "Y[i, k] = X[i, j] * W[j, k]"
    parse_tree = parse(op)
    result = transform(parse_tree)
    assert isinstance(result, Statement)
