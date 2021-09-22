'''Tests for the second coursework Question 4'''
import pytest
import numpy as np
from scipy.linalg import solve_triangular
import cla_utils as cla

CID = 1357062
  
# Test the solution for Ax = b produced by LUsolveA for random float arrays 
@pytest.mark.parametrize('m', [5, 10, 20, 30, 40])
def test_LUsolveq2A(m):
    np.random.seed(CID+m)
    A = np.random.random((m,m))
    x0 = np.random.random(m)
    b = A@x0
    # Create matrix M to precondition against
    M = np.triu(A)
    apply_pc = lambda x: solve_triangular(M, x)
    x, nits, r = cla.GMRES(A, b, 50, 10**-10, return_residual_norms=True, 
                           apply_pc=apply_pc)
    assert(np.linalg.norm(apply_pc(A@x0-b) <= 10**-10))
