'''Tests for the second coursework Question 2'''
import pytest
import numpy as np
import q2

CID = 1357062
  
# Test the solution for Ax = b produced by LUsolveA for random float arrays 
@pytest.mark.parametrize('m', [5, 11, 50, 100, 501, 1000])
def test_LUsolveq2A(m):
    np.random.seed(CID+m)
    rand = np.random.random(2 + m)
    c = rand[0]
    d = rand[1]
    A = c*np.eye(m) + d*(np.eye(m,k=1)+np.eye(m,k=-1))
    A[0, -1] = d
    A[-1, 0] = d
    b = rand[2:]
    x = q2.LUsolveq2A(c, d, b, copy=True)
    b.shape=(m, b.size//m)
    assert(np.allclose(A@x, b))

# Test the solution for Ax = b produced by LUsolveA for random integer arrays        
@pytest.mark.parametrize('m', [5, 11, 50, 100, 501, 1000])
def test_LUsolveA_int(m):
    np.random.seed(CID+m)
    rand = np.random.randint(1, 1000, 2 + m)
    c = rand[0]
    d = rand[1]
    A = c*np.eye(m) + d*(np.eye(m,k=1)+np.eye(m,k=-1))
    A[0, -1] = d
    A[-1, 0] = d
    b = rand[2:]
    x = q2.LUsolveq2A(c, d, b, copy=True)
    b.shape=(m, b.size//m)
    assert(np.allclose(A@x, b))

# Test the solution for Ax = b produced by LUsolveA for random float arrays 
@pytest.mark.parametrize('m', [5, 11, 50, 100, 501, 1000])
def test_solve_inplace(m):
    np.random.seed(CID+m)
    rand = np.random.random(2 + m)
    c = rand[0]
    d = rand[1]
    A = c*np.eye(m) + d*(np.eye(m,k=1)+np.eye(m,k=-1))
    A[0, -1] = d
    A[-1, 0] = d
    A0 = np.array(A)
    b = rand[2:]
    b0 = np.array(b)
    x = q2.solve_inplace(A0, b0)
    b.shape=(m, b.size//m)
    assert(np.allclose(A@x, b))


# Test wave equation for u = sin(2pi*x)cos(2pi*t)
@pytest.mark.parametrize('M, tsteps, dt', [(100, 500, 0.0001),
                                           (200, 500, 0.001),
                                           (200, 1000, 0.001)])
def test_solve_wave_equ_sincos(M, tsteps, dt):
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 0
    u = q2.solve_wave_equ(u0, u1, M, tsteps, dt)
    x = np.linspace(1.0/M, 1, M)
    t = dt * tsteps
    sol = np.sin(2*np.pi*x) * np.cos(2*np.pi*t)
    sol_minus_u = sol-u
    assert(max(abs(sol_minus_u))<10**-3)
    assert(np.linalg.norm(sol_minus_u)<10**-2)

# Test wave equation for u = sin(2pi*x)sin(2pi*t)   
@pytest.mark.parametrize('M, tsteps, dt', [(100, 500, 0.0001),
                                           (200, 500, 0.001),
                                           (200, 1000, 0.001)])
def test_solve_wave_equ_sinsin(M, tsteps, dt):
    u0 = lambda x: 0
    u1 = lambda x: 2*np.pi*np.sin(2*np.pi*x)
    u = q2.solve_wave_equ(u0, u1, M, tsteps, dt)
    x = np.linspace(1.0/M, 1, M)
    t = dt * tsteps
    sol = np.sin(2*np.pi*x) * np.sin(2*np.pi*t)
    sol_minus_u = sol-u
    assert(max(abs(sol_minus_u))<10**-3)
    assert(np.linalg.norm(sol_minus_u)<10**-2)
 
# Test wave equation for u = sin(2pi(x+t))  
@pytest.mark.parametrize('M, tsteps, dt', [(100, 500, 0.0001),
                                           (200, 500, 0.001),
                                           (200, 1000, 0.001)])
def test_solve_wave_equ_sin(M, tsteps, dt):
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 2*np.pi*np.cos(2*np.pi*x)
    u = q2.solve_wave_equ(u0, u1, M, tsteps, dt)
    x = np.linspace(1.0/M, 1, M)
    t = dt * tsteps
    sol = np.sin(2*np.pi*(x+t))
    sol_minus_u = sol-u
    assert(max(abs(sol_minus_u))<10**-3)
    assert(np.linalg.norm(sol_minus_u)<10**-2)
