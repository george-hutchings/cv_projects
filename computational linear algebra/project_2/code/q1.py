import numpy as np

def LUsolveA(c, d, b, copy=False):
    '''
    Given c, d which represent the tridiagonal matrix A (mxm), as described in 
    question 1, and b, solves the equation Ax = b, by LU decomposition and taking 
    advantage of the form of A.

    Parameters
    ----------
    c : Float
        Represents the diagonal entries of A (mxm).
    d : Float
        Represents the sub- and super-diagonal entries of A (mxm).
    b : mxn real numpy array, or 1D m-length real numpy array.
        b as in the equation Ax = b.
    copy : Bool, optional
        Deterimnes whether the algorithm should be performed in-place, if False 
        algorithm will be performed in-place
        The default is False.

    Returns
    -------
    b : mxn real numpy array (or mx1 if b was in putted as 1 dimensional)
        Solution to Ax = b.

    '''
    if np.can_cast(b.dtype, float):
        dtype='float64'
        b = b.astype(dtype, copy=copy)
    else:
        dtype='complex128'
    m = b.shape[0]
    # Convert b to a mx1 matrix if it is an array to allow 2D slice notation.
    if b.ndim == 1: 
        b.shape = (m, 1)
    U = np.zeros(m, dtype=dtype) # Initialise array containing diagonal values of U.
    U[0] = c
    for k in range(m-1): # LU factorise and forward substitute.
        l = d / U[k]
        b[k+1, :] -= l * b[k, :]
        U[k+1] = U[0] - l*d
    b[-1, :] /= U[-1]
    for k in range(m-2, -1, -1): # Back substitute.
        b[k, :]= (b[k, :] - d*b[k+1, :]) / U[k]
    return b