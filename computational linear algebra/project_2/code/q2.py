import numpy as np
import q1
import timeit
import cla_utils as cla
import matplotlib.pyplot as plt


def LU_banded(A, bw, copy=False):
    '''
    Given a banded matrix A, with upper and lower bandwidth = bw, the LU 
    factorisation is computed, taking advantage of the bandwidth.

    Parameters
    ----------
    A : mxm real numpy array of tridiagonal form as described in Question 1, 
    with diagonal assumed non zero.
    bw : Integer
        The upper and lower bandwidth of the matrix A.
    copy : bool, optional
        Deterimnes whether the algorithm should be performed in-place, if False 
        algorithm will be performed in-place
        The default is False.

    Returns
    -------
    A : mxm numpy array
        Such that upper triangular part of A represents U and the lower 
        triangular (not including the diagonal) represents L (which has 1s on 
        the diagonal) where A (input) = LU

    '''
    if copy:
        A = np.array(A)
        
    if np.issubdtype(A[0,0], int):
        A = A.astype(float, copy=False)
        
    _, m = A.shape
    # No need for minimum (as used in the lecture notes) since we cannot 
    # 'over index' arrays in numpy.
    for k in range(m-1):
        k1 = k + 1
        k2 = k1 + bw
        A[k1:k2, k] /= A[k, k]
        A[k1:k2, k1:k2] -= np.outer(A[k1:k2, k],A[k, k1:k2])
    return A


# Written as one function as to ensure that both LU_banded and cla.LU_inplace 
# are being tested on identitcal matrices.
def timing_LU_comparison(repeats, m):
    '''
    Function to compare the time required to LU factorise A, which is of 
    tridiagonal form, using LU_banded and LU_inplace.

    Parameters
    ----------
    repeats : Integer
        Number of times to repeat the timing.
    m : Integer
        Size of matrices to time on (mxm).

    Returns
    -------
    None.

    '''
    times = np.zeros((repeats, 2))
    for i in range(repeats):
        np.random.seed(repeats+m+i)
        rand = np.random.random(2)
        c = rand[0]
        d = rand[1]
        A = c*np.eye(m) + d*(np.eye(m,k=1)+np.eye(m,k=-1))
        A[0, -1] = d
        A[-1, 0] = d
        A0 = np.array(A)
        times[i, 0] = timeit.timeit(lambda: LU_banded(A, bw=m-1), number=1)
        times[i, 1] = timeit.timeit(lambda: cla.LU_inplace(A0), number=1)
    means = [np.mean(times[:, 0]), np.mean(times[:, 1])]
    percentdiff = 100 * means[0] / means[1] - 100
    print('Avg (%dx) percentage difference of computational time of LU_banded'
          ' against LU_inplace on %dx%d matrices is %.2f%%' % 
          (repeats, m, m, percentdiff))
    return



def inv2by2(A):
    '''
    Finds the inverse of A, a 2x2 invertible matrix.

    Parameters
    ----------
    A : 2x2 real numpy array
        Matrix to find the inverse of.

    Returns
    -------
    Ainv : 2x2 real numpy array
        Inverse of A.

    '''
    Ainv = np.array(A[::-1, ::-1].T) # Transpose along the anti-diagonal 
    Ainv[(1,0), (0,1)] = -Ainv[(1,0), (0,1)]
    Ainv /= Ainv[0,0]*Ainv[1,1] - Ainv[1,0]*Ainv[0,1]
    return Ainv


def LUsolveq2A(c, d, b, copy=False):
    '''
    Given c, d which represent the matrix A (mxm), as described in 2b, and b, 
    solves the equation Ax = b, by LU decomposition and taking advantage of the
    form of A.

    Parameters
    ----------
    c : Float
        Represents the diagonal entries of A (mxm).
    d : Float
        Represents the sub- and super-diagonal entries of A (mxm), as well as 
        its top
        right and bottom left entries.
    b : mxn real numpy array, or 1D m-length real numpy array.
        b as in the equation Ax = b.
    copy : Bool, optional
        Deterimnes whether the algorithm should be performed in-place, if False 
        algorithm will be performed in-place
        The default is False.

    Returns
    -------
    Ainvb : mxn real numpy array (or mx1 if b was in putted as 1 dimensional)
        Solution to Ax = b.

    '''
    if np.can_cast(b.dtype, float):
        dtype='float64'
        b = b.astype(dtype, copy=copy)
    else:
        dtype='complex128'
    if b.ndim == 1:
        m, n = len(b), 1
        b.shape = (m, n)
    else:
        m, n = b.shape
    # Concatenate Uhat and b to solve Tx = Uhat and Ty = b simultaneously 
    Uhatb = np.zeros((m,2+n),dtype=dtype) 
    Uhatb[:,2:] = b
    Uhatb[0, 0] = d
    Uhatb[-1, 1] = 1
    TinvUhatb = q1.LUsolveA(c, d, Uhatb, copy=False)
    TinvUhat, Tinvb = TinvUhatb[:, :2], TinvUhatb[:, 2:]
    # Take advantage of the sparsity of Vhat hence we can multiply by hand to
    # reduce operation count.
    IminusVhatTinvUhat = np.zeros((2,2), dtype=dtype)
    IminusVhatTinvUhat[0,0] = TinvUhat[-1, 0] + 1
    IminusVhatTinvUhat[0,1] = TinvUhat[-1, 1]
    IminusVhatTinvUhat[1,0] = TinvUhat[0, 0]*d
    IminusVhatTinvUhat[1,1] = TinvUhat[0, 1]*d + 1
    inv = inv2by2(IminusVhatTinvUhat) 
    VhatTinvb = np.zeros((2,1), dtype=dtype)
    VhatTinvb[0,0] = Tinvb[-1]
    VhatTinvb[1,0] = Tinvb[0] * d
    Ainvb = Tinvb - TinvUhat@(inv@VhatTinvb)
    return Ainvb



def solve_inplace(A, b):
    '''
    Function to solve Ax=b, by using inplace methods.

    Parameters
    ----------
    A : mxm numpy 
        mxm matrix as part of the system Ax=b.
    b : m length 1D numpy array
        m lenght vector as part of the system Ax=b.

    Returns
    -------
    b : m length 1D numpy array
        solution to Ax=b.
    '''
    A = cla.LU_inplace(A)
    L = np.array(A)
    np.fill_diagonal(L, 1)
    b = cla.solve_L(L, b)
    b = cla.solve_U(A, b)
    return b
    

def timing_solve_comparison(repeats, m):
    '''
    Function to compare the time required to solve the system Ax = b, (for 
    random mxm A of form given in the question using LU_banded and LU_inplace.

    Parameters
    ----------
    repeats : Integer
        Number of times to repeat the timing.
    m : Integer
        Size of matrices to time on (mxm).

    Returns
    -------
    None.

    '''
    times = np.zeros((repeats, 2))
    for i in range(repeats):
        np.random.seed(repeats+m+i)
        rand = np.random.random(2+m)
        c = rand[0]
        d = rand[1]
        A = c*np.eye(m) + d*(np.eye(m,k=1)+np.eye(m,k=-1))
        A[0, -1] = d
        A[-1, 0] = d
        A0 = np.array(A)
        b = rand[2:]
        b0 = np.array(b)
        times[i, 0] = timeit.timeit(lambda: LUsolveq2A(c,d, b,), number=1)
        times[i, 1] = timeit.timeit(lambda: solve_inplace(A0, b0), number=1)
    means = [np.mean(times[:, 0]), np.mean(times[:, 1])]
    percentdiff = 100 * means[0] / means[1] - 100
    print('Avg (%dx) percentage difference of computational time of LUsolveq2A'
          ' against solve_inplace on %dx%d matrices is %.2f%%' % 
          (repeats, m, m, percentdiff))
    return



def solve_wave_equ(u0, u1, M, tsteps, dt, interval=0, save=False, plot=False, filename=None):
    '''
    Solves the wave equation as presented in Question 2.

    Parameters
    ----------
    u0 : Function
        u0(x), initial function of u(x) at time=0, as defined in the question.
    u1 : Function
        u1(x), initial function of w(x) at time=0, as defined in the question.
    M : Integer
        How many points to discretise x into, ie delta x := 1/M.
    tsteps : Integer
        How many timesteps to perform.
    dt : Float
        The size of a single timestep.
    interval : Integer, optional
        A save/plot will occur every interval number of timesteps ( 0 means
        will save every timestep). The default is 0.
    save : Boolean, optional
        If True will save to the disk every interval timesteps. The default
        is False.
    plot : Boolean, optional
        If True will plot u every interval timesteps. The default is False.

    Returns
    -------
    un : 1D numpy array of length M
        The solution evaluated at x=1/M, 2/M,..., 1 and t = tsteps*dt

    '''
    if filename == None:
        filename = 'q2output'
    # Initialising Values
    x = np.linspace(1.0 / M, 1, M)
    un = u0(x)
    wn = u1(x)
    dt_2 = 0.5 * dt
    C1 = (dt_2*M)**2
    c = 1 + 2*C1
    d = -C1
    dtMM = dt*M*M
    
    # Initial Save/Plot (of u0).
    if save:
        file = open("Outputs/%s.txt" % (filename), "wb")
        np.savetxt(file, un, newline =' ')
        file.write(b"\n")  
    if plot:
        c_m = plt.cm.viridis
        plt.plot(x, un, color=c_m(0))
        
    for i in range(1, tsteps+1):
        fi = (np.roll(un, 1) + np.roll(un, -1) -2*un) * dtMM
        fi += (np.roll(wn, 1) + np.roll(wn, -1) -2*wn) * C1
        fi += wn
        wnplus1 = LUsolveq2A(c, d, fi, copy=False)[:, 0]
        un += dt_2*(wn + wnplus1)
        wn = wnplus1
        
        if interval == 0 or i%interval == 0:
            if save:
                np.savetxt(file, un, newline =' ')
                file.write(b"\n")
            if plot:
                plt.plot(x, un, color=c_m(i/tsteps))               
    if save:
        file.close()
    if plot:
        norm = plt.Normalize(0, tsteps*dt)
        sm = plt.cm.ScalarMappable(cmap=c_m, norm=norm)
        clb = plt.colorbar(sm)
        clb.ax.invert_yaxis()
        clb.ax.set_title('Time')
        plt.title(label = 'Approximate solution to the wave equation (Q2)')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.tight_layout()
        plt.savefig('Figures/%s.png' % (filename),dpi=600)
        plt.show()
    return un


if __name__ == "__main__":
    
    timing_LU_comparison(200,100) 
    timing_LU_comparison(200,200)
    timing_LU_comparison(200,300)
    
    timing_solve_comparison(200,100) 
    timing_solve_comparison(200,200)
    timing_solve_comparison(200,300)
    
    # u = sin(2pi*x)sin(2pi*t)
    u0 = lambda x: 0 * x
    u1 = lambda x: 2*np.pi*np.sin(2*np.pi*x)
    solve_wave_equ(u0, u1, M=200, tsteps=1000, dt=0.001, interval=10,
                   plot=True, filename='sinsin')
    
    # u = sin(2pi*x)cos(2pi*t)
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 0 * x
    solve_wave_equ(u0, u1, M=200, tsteps=1000, dt=0.001, interval=10,
                   plot=True, filename='sincos')
    
    # u = sin(2pi(x+t))
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 2*np.pi * np.cos(2*np.pi*x)
    solve_wave_equ(u0, u1, M=200, tsteps=500, dt=0.001, interval=5,
                   plot=True, filename='sin')

