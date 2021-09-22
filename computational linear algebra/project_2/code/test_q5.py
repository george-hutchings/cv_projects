'''Tests for the second coursework Question 5'''
import pytest
import numpy as np
import q5

CID = 1357062
maxit=20
tol = 10**-10
a = 0.001
# Test wave equation for u = sin(2pi*x)cos(2pi*t)
@pytest.mark.parametrize('M, N, dt', [(100, 500, 0.001), (200, 300, 0.001),
                                           (200, 500, 0.002)])
def test_solve_wave_equ_5sincos(M, N, dt):
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 0
    nits, Uk = q5.solve_wave_equ5(u0, u1, M, N, dt, a, maxit=maxit, tol=tol)
    assert(nits != -1)
    index = M * (N-1)*2
    u = Uk[index:index+M]
    x = np.linspace(1.0/M, 1, M)
    t = dt * N
    sol = np.sin(2*np.pi*x) * np.cos(2*np.pi*t)
    sol_minus_u = sol-u
    assert(max(abs(sol_minus_u))<10**-3)
    assert(np.linalg.norm(sol_minus_u)<10**-2)


# Test wave equation for u = sin(2pi*x)sin(2pi*t)    
@pytest.mark.parametrize('M, N, dt', [(100, 500, 0.001), (200, 300, 0.001),
                                           (200, 500, 0.002)])
def test_solve_wave_equ_5sinsin(M, N, dt):
    u0 = lambda x: 0
    u1 = lambda x: 2*np.pi*np.sin(2*np.pi*x)
    nits, Uk = q5.solve_wave_equ5(u0, u1, M, N, dt, a, maxit=maxit, tol=tol)
    assert(nits != -1)
    index = M * (N-1)*2
    u = Uk[index:index+M]
    x = np.linspace(1.0/M, 1, M)
    t = dt * N
    sol = np.sin(2*np.pi*x) * np.sin(2*np.pi*t)
    sol_minus_u = sol-u
    assert(max(abs(sol_minus_u))<10**-3)
    assert(np.linalg.norm(sol_minus_u)<10**-2)
 
# Test wave equation for u = sin(2pi(x+t))    
@pytest.mark.parametrize('M, N, dt', [(100, 500, 0.001), (200, 300, 0.001),
                                           (200, 500, 0.002)])
def test_solve_wave_equ_5sin(M, N, dt):
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 2*np.pi*np.cos(2*np.pi*x)
    nits, Uk = q5.solve_wave_equ5(u0, u1, M, N, dt, a, maxit=maxit, tol=tol)
    assert(nits != -1)
    index = M * (N-1)*2
    u = Uk[index:index+M]
    x = np.linspace(1.0/M, 1, M)
    t = dt * N
    sol = np.sin(2*np.pi*(x+t))
    sol_minus_u = sol-u
    assert(max(abs(sol_minus_u))<10**-3)
    assert(np.linalg.norm(sol_minus_u)<10**-2)
