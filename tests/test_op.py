import numpy as np
import pytest

from dldsl.compile import op
from dldsl.errors import AxisDisagreement, UnexpectedTensor, MissingTensor


BATCH_SIZE = 32
SEQ_LENGTH = 16
HIDDEN_DIM = 128
OUTPUT_DIM = 64

def test_negation():
    X = np.random.randn(HIDDEN_DIM)
    negate = op("Y[n] = -X[n]")
    result = negate(X=X)
    assert np.allclose(result, -X)


def test_matmul_transform():
    X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    W = np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
    matmul = op("Y[i, k] = X[i, j] * W[j, k]")
    result = matmul(X=X, W=W)
    assert np.allclose(result, np.matmul(X, W))


def test_broadcast_add():
    A = np.random.randn(2)
    B = np.random.randn(3)
    expected = np.expand_dims(A, 1) + np.expand_dims(B, 0)
    broadcast_add = op("Y[a, b] = A[a] + B[b]")
    result = broadcast_add(A=A, B=B)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_axis_disagreement():
    with pytest.raises(AxisDisagreement):
        X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
        W = np.random.randn(OUTPUT_DIM, OUTPUT_DIM)
        matmul = op("Y[i, k] =  X[i, j] * W[j, k]")
        result = matmul(X=X, W=W)


def test_unexpected_tensor():
    with pytest.raises(UnexpectedTensor):
        X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
        W = np.random.randn(OUTPUT_DIM, OUTPUT_DIM)
        matmul = op("Y[i, k] =  X[i, j] * W[j, k]")
        result = matmul(A=X)


def test_missing_tensor():
    with pytest.raises(MissingTensor):
        X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
        matmul = op("Y[i, k] =  X[i, j] * W[j, k]")
        result = matmul(X=X)


def test_y_equals_wx_plus_b():
    X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    W = np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
    B = np.random.randn(OUTPUT_DIM)
    wx_plus_b = op("Y[i, k] = W[j, k] * X[i, j] + B[k]")
    result = wx_plus_b(W=W, X=X, B=B)
    expected = np.matmul(X, W) + B
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_negation_subtraction():
    X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    W = np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
    B = np.random.randn(OUTPUT_DIM)
    wx_plus_b = op("Y[i, k] = -W[j, k] * X[i, j] - B[k]")
    result = wx_plus_b(W=W, X=X, B=B)
    expected = np.matmul(-X, W) - B
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_simple_parens():
    ExampleBias = np.random.randn(BATCH_SIZE)
    Input = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    compiled = op("Output[batch, hidden] = (Input[batch, hidden] + ExampleBias[batch])")
    result = compiled(Input=Input, ExampleBias=ExampleBias)
    expected = Input + np.expand_dims(ExampleBias, 1)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_proper_bcast_reduce():
    Input = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    ExampleBias = np.random.randn(BATCH_SIZE)
    OutputBias = np.random.randn(HIDDEN_DIM)
    compiled = op("Output[batch, hidden] = Input[batch, hidden] + ExampleBias[batch] + OutputBias[hidden]")
    result = compiled(Input=Input, ExampleBias=ExampleBias, OutputBias=OutputBias)
    expected = Input + np.expand_dims(ExampleBias, 1) + np.expand_dims(OutputBias, 0)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_parens():
    ExampleBias = np.random.randn(BATCH_SIZE)
    Input = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    Weight = np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
    compiled = op(
        "Output[batch, output] = "
        "(Input[batch, hidden] + ExampleBias[batch]) * Weight[hidden, output]"
    )
    result = compiled(Input=Input, ExampleBias=ExampleBias, Weight=Weight)

    simpler_compiled = op(
        "Output[batch, output] = Input[batch, hidden] * Weight[hidden, output]"
    )
    simpler_result = simpler_compiled(Input=Input + np.expand_dims(ExampleBias, 1), Weight=Weight)

    assert np.allclose(result, simpler_result)

    expected = np.matmul(Input + np.expand_dims(ExampleBias, 1), Weight)
   
    manual = np.zeros((BATCH_SIZE, OUTPUT_DIM))
    for batch in range(BATCH_SIZE):
        for hidden in range(HIDDEN_DIM):
            for output in range(OUTPUT_DIM):
                manual[batch, output] += (Input[batch, hidden] + ExampleBias[batch]) * Weight[hidden, output]

    assert result.shape == expected.shape
    assert np.allclose(manual, expected)
    assert np.allclose(result, manual)   
    assert np.allclose(result, expected)


def test_more_complex():
    ExampleBias = np.random.randn(BATCH_SIZE)
    Input = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    Weight = np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
    OutputBias = np.random.randn(OUTPUT_DIM)
    compiled = op(
        "Output[batch, output] = "
        "(Input[batch, hidden] + ExampleBias[batch]) * Weight[hidden, output] + OutputBias[output]"
    )
    result = compiled(Input=Input, ExampleBias=ExampleBias, Weight=Weight, OutputBias=OutputBias)
    expected = np.matmul(Input + np.expand_dims(ExampleBias, 1), Weight) + np.expand_dims(OutputBias, 0) 
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_matrix_decomp():
    SMALLER_DIM = 2
    X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    W1 = np.random.randn(HIDDEN_DIM, SMALLER_DIM)
    W2 = np.random.randn(SMALLER_DIM, HIDDEN_DIM)
    compiled = op(
        "Output[batch, output] = X[batch, hidden] * W1[hidden, smaller] * W2[smaller, output]"
    )
    result = compiled(X=X, W1=W1, W2=W2)
    expected = np.matmul(np.matmul(X, W1), W2)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_empty_sum():
    x = np.random.randn(5, 7)
    compiled = op("Output[] = x[a, b]")
    result = compiled(x=x)
    expected = np.sum(x, keepdims=False)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)
