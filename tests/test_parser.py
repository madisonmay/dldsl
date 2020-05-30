from lark import Tree, Token

from dldsl import parse


def test_matmul_parse():
    op = "Y[i, k] = W[i, j] * X[j, k]"
    result = parse(op)
    assert isinstance(result, Tree)


def test_more_complex_statement():
    op = "Y[batch, out] = W[hidden, out] * X[batch, hidden]  + B[out]"
    result = parse(op)
    assert isinstance(result, Tree)


def test_outer_addition():
    op = "Y[a, b] = A[a] + B[b]"
    result = parse(op)
    assert isinstance(result, Tree)
