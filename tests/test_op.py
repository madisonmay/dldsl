import numpy as np
import pytest

from dldsl.compile import op
from dldsl.errors import AxisDisagreement, UnexpectedTensor, MissingTensor


BATCH_SIZE = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 64

def test_matmul_transform():
    X = np.random.randn(BATCH_SIZE, HIDDEN_DIM)
    W = np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
    matmul = op("Y[i, k] =  X[i, j] * W[j, k]")
    result = matmul(X=X, W=W)


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
