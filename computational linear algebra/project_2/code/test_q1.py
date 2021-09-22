'''Tests for the second coursework Q1'''
import pytest
import numpy as np
import q1 

CID = 1357062
  
# Test the solution for Ax = b produced by LUsolveA for random float 1D arrays 
@pytest.mark.parametrize('m', [5, 11, 53, 100, 501, 1000])
def test_LUsolveA(m):
    np.random.seed(CID+m)
    rand = np.random.random(2+m)
    b = rand[2:]
    x = q1.LUsolveA(rand[0], rand[1], b, copy=True)
    b.shape=(m, b.size//m)
    A = rand[0]*np.eye(m) + rand[1]*(np.eye(m,k=1)+np.eye(m,k=-1))
    assert(np.allclose(A@x, b))

# Test the solution for Ax = b produced by LUsolveA for random integer 1D arrays        
@pytest.mark.parametrize('m', [5, 11, 53, 100, 501, 1000])
def test_LUsolveA_int(m):
    np.random.seed(CID+m)
    rand = np.random.randint(1, 1000, 2+m)
    b = rand[2:]
    x = q1.LUsolveA(rand[0], rand[1], b, copy=True)
    b.shape=(m, b.size//m)
    A = rand[0]*np.eye(m) + rand[1]*(np.eye(m,k=1)+np.eye(m,k=-1))
    assert(np.allclose(A@x, b))
            
# Test the solution for Ax = b produced by LUsolveA for random float 2D arrays 
@pytest.mark.parametrize('m', [5, 11, 53, 100, 501, 1000])
def test_LUsolveA_array(m):
    np.random.seed(CID+m)
    rand = np.random.random(2)
    b = np.random.random((m,m//2))
    x = q1.LUsolveA(rand[0], rand[1], b, copy=True)
    A = rand[0]*np.eye(m) + rand[1]*(np.eye(m,k=1)+np.eye(m,k=-1))
    assert(np.allclose(A@x, b))

# Test the solution for Ax = b produced by LUsolveA for random integer 2D arrays        
@pytest.mark.parametrize('m', [5, 11, 53, 100, 501, 1000])
def test_LUsolveA_array_int(m):
    np.random.seed(CID+m)
    rand = np.random.randint(1, 1000, m)
    b = np.random.randint(1, 1000, (m,m//2))
    x = q1.LUsolveA(rand[0], rand[1], b, copy=True)
    A = rand[0]*np.eye(m) + rand[1]*(np.eye(m,k=1)+np.eye(m,k=-1))
    assert(np.allclose(A@x, b))
