# Deep Learning Domain Specific Language

 
 Define operations that take in tensors and output tensors using a syntax similar to "einsum".


# Motivation

Einsum is an expressive way to denote a wide range of linear algebra operations that
correspond to sum reductions over tensor products.

The `dldsl` package intends to introduce a new einsum-like syntax capable of expressing a broader
ranger of operations than einsum alone.

To illustrate, the computation beformed by the following einsum call:
```
tf.einsum(`ijk,ikl->ijl`, A, B)
```

is equivalent to the following expansion:
```python
# First we construct an output of the provided shape
output = np.zeros(i, j, l)

# Then, we use nested for loops over 
# the indices present on the right
# In our case, this is (i, j, l)
for _i in range(i):
    for _j in range(j):
        for _l in range(l):
            # For the remaining axes, we use an 
            # intermediate variable to compute 
            # a sum reduction over that axis
            total = 0
            for _k in range(k):
                # In the inner-most loop,
                # we multiply the indices indicated by the 
                # left-hand side of the equation
                #
                # Our tensor "operands" are the elements on the 
                # right hand side of the equation
                total += A[i, j, k] * B[i, k, l]
            # And finally, set the value at the 
            # location indicated by the right
            output[i, j, l] = total
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

### Trace
```
Y = W[i] * W[i]
# W = np.random.rand(i)
# Y = np.trace(W)
``` 

### Or something arbitrarily complexB
```
Y[batch, hidden] = X[batch, hidden] * W[hidden, output] + B[output]
```


# Usage:
```python
import dldsl as dl

op = dl.compile(
    "Y[batch, hidden] = X[batch, hidden] * W[hidden, output] + B[output]"
)
Y = op(X=X, W=W, B=B) 
```
