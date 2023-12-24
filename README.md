### WARNING: 
I have no idea what I'm doing but had fun with this regardless.  
I've made no attempts to make this fast and currently only uses numpy as a backend.
Don't actually use this please.

# Motivation

Einsum is an expressive way to denote a wide range of linear algebra operations that
correspond to sum reductions over tensor products.

The `dldsl` package intends to introduce a new einsum-like syntax capable of expressing a broader
ranger of operations than einsum alone.

To illustrate, the computation peformed by the following einsum call:
```
tf.einsum(`ijk,ikl->ijl`, A, B)
```

is equivalent to the following expansion:
```python
output = np.zeros(I, J, L)
for i in range(I):
    for j in range(J):
        for k in range(K):
            for l in range(L):
                output[i, j, l] += A[i, j, k] * B[i, k, l]
```

The proposed-syntax of DLDSL allows for generic scalar math to be performed on the operands
of the einsum operation.

### Matrix multiplication
 ```
Y[i, k] = W[i, j] * X[j, k]
# W = np.random.rand(i, j)
# X = np.random.rand(j, k)
# Y = np.matmul(W, X)
 ```

### Broadcast addition
```
Y[i, j] = A[i] + B[j]
# A = np.random.rand(i)
# B = np.random.rand(k)
# Y = np.expand_dims(A, 1) + np.expand_dims(1, B)
```

### Or something arbitrarily complex
```
Output[batch, output] = (Input[batch, hidden] + ExampleBias[batch]) * Weight[hidden, output] + OutputBias[output] 
```


# Usage:
```python
import dldsl as dl

op = dl.compile(
    "Y[batch, output] = X[batch, hidden] * W[hidden, output] + B[output]"
)
Y = op(X=X, W=W, B=B) 
```
