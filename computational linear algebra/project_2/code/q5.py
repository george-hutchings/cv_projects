import numpy as np
import matplotlib.pyplot as plt
import q2


def solve_wave_equ5(u0, u1, M, N, dt, a, maxit, tol=10**-10, interval=0, 
                   plot=False, filename=None, U0=None):
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
    N : Integer
        How many timesteps to perform.
    dt : Float
        The size of a single timestep.
    maxit : Integer
        The maximum number of iterations.
    a : Float
        Alpha > 0, as defined in Question 5, must be sufficently small for 
        convergence of the solution.
    tol: Float, optional
        Determines the maximum error for which (the largest difference between
        two solutions should be considered equal, allowing for a termination 
        before maxit has been reached. The default is 10^-4.
    interval : Integer, optional
        A plot will occur every interval number of timesteps ( 0 means
        will save every timestep). The default is 0.
    plot : Boolean, optional
        If True will plot u every interval timesteps. The default is False.
    U0 : 2MN vector
        the tinitial guess for the Solution

    Returns
    -------
    Uk : 1D numpy array of length 2MN
        As defined in the Question, (p_1, q_1, ... , p_N, q_N), evaluated at 
        x=1/M, 2/M,..., 1 and t = (dt, 2dt, ... Ndt).
    nits : Integer
        If converged, the number of iterations required, otherwise equal to -1.

    '''
    
    if filename == None:
        filename = 'q3output'
        
    # Initial Values
    x = np.linspace(1.0 / M, 1, M)
    pn = u0(x)
    qn = u1(x)
    
    # Initial Plot (of u0).
    if plot:
        c_m = plt.cm.viridis
        plt.plot(x, pn, color=c_m(0))
    
    
    # Create D_1 D_2, diagonal matrices, they are represented by just their 
    # diagonals.
    zerotoNminus1 = np.arange(N)
    D1diag = -2j* np.pi * zerotoNminus1 #2pi N ij
    D1diag /= N
    D1diag = np.exp(D1diag)
    D1diag *= a**(1/N)
    D2diag = np.array(D1diag)
    D1diag = 1 - D1diag
    D2diag = 0.5 * (1 + D2diag)
    D1overD2D2 = D1diag / (D2diag*D2diag)
    
    
    # Useful values to save, calculating them prior avoids having to calculate 
    # them repeatedly in loops.
    dt_dxdx = dt * M * M
    dtdt_dxdx = dt * dt_dxdx
    twodtdt_dxdx = 2 * dtdt_dxdx
    M2 = 2 * M
    MN2 = M2 * N
    r = np.zeros(M2)
    Cs = D1diag*D1overD2D2 + twodtdt_dxdx 
    d = -dtdt_dxdx
    Index1s = M2*zerotoNminus1
    Index2s = Index1s + M
    Index3s = Index2s + M
    D2dt = D2diag*dt
    
    # Create Dhat and Dhat inverse, that is the matricies used to  form V^-1 
    # and V, not that they have been made as Mx1 matrices, containing only the 
    # diagonals since all other values are zero.
    Dhatinv = np.zeros(N)
    Dhatinv = a**(-zerotoNminus1/N)
    Dhatinv *= N
    # Forming them as Mx1 matrices to allow for broadcasting.
    Dhatinv = Dhatinv[:,None] 
    Dhat = 1 / Dhatinv
    
    # Function equivalent to matrix multiplication on the left by B_21.
    B21 = lambda x: dt_dxdx*(2*x - (np.roll(x, 1)+np.roll(x, -1)))
    
    # Form initial r
    r[:M] = pn
    r[:M] += 0.5*dt*qn
    r[M:] = qn
    r[M:] -= 0.5*B21(pn)
    
    # Preallocate Arrays
    if U0 is None:
        Uk = np.zeros(MN2, dtype=complex)
    else:
        Uk = U0
    R = np.zeros(MN2, dtype=complex)
    Ukminus1 = np.array(Uk) 
    Bpkqk = np.zeros(M2, dtype=complex)
    
        
    for k in range(maxit):
        
        # Creating R
        pkqk = Uk[M2 * (N-1):]
        Bpkqk[:M] = -dt * pkqk[M:]
        Bpkqk[M:] = B21(pkqk[:M])
        Bpkqk = 0.5*Bpkqk - pkqk
        R[:M2] = r + a*Bpkqk
        R[M2:] = 0
        
        # Step i 
        R.shape = (N, M2)
        R = np.fft.fft(Dhat*R, axis=0)
        R = R.reshape(MN2)
        
        # Step ii
        for i in range(N):
            d1k = D1diag[i]
            d2k = D2diag[i]
            d1k_d2kd2k = D1overD2D2[i]
            c = Cs[i]
            index1 = Index1s[i]
            index2 = Index2s[i]
            index3 = Index3s[i]
            rhat1 = R[index1:index2]
            rhat2 = R[index2:index3]
            rhat = rhat2*d1k_d2kd2k - (B21(rhat1)/d2k)
            qn = q2.LUsolveq2A(c, d, rhat)[:, 0]
            pn = (rhat1 + D2dt[i]*qn)/d1k
            Uk[index1:index2] = pn
            Uk[index2:index3] = qn

        # Step iii
        Uk.shape = (N, M2)
        Uk = np.fft.ifft(Uk, axis=0) 
        Uk = Dhatinv*Uk
        Uk = Uk.reshape(MN2)

        # Terminate loop if within tolerance.
        # Not using norm since this increases with the size of the vector and 
        # our Uk is large (2MN) .
        error = max(abs(Ukminus1 - Uk)) 
        if error < tol:
            maxit = k+1
            break
        Ukminus1 = np.array(Uk)
    if error >= tol:
        maxit=-1
        
    Uk = np.real(Uk) # Convert to real to avoid warnings when plotting. 
    if interval == 0:
        saves = np.arange(N)
    elif interval < 0:
        saves = []
    else:
        saves = np.arange(interval-1,N,interval) 
    for i in saves:
        if plot:
            Mi = M*i*2
            pn = Uk[Mi:Mi+M]
            plt.plot(x, pn, color=c_m(i/N))               
    if plot:
        norm = plt.Normalize(0, N*dt)
        sm = plt.cm.ScalarMappable(cmap=c_m, norm=norm)
        clb = plt.colorbar(sm)
        clb.ax.invert_yaxis()
        clb.ax.set_title('Time')
        plt.title(label = 'Approximate solution to the wave equation (Q5)')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.tight_layout()
        plt.savefig('Figures/%s.png' % (filename),dpi=600)
        plt.show()
    return maxit, Uk

       
if __name__ == "__main__":
    
    # u = sin(2pi*x)sin(2pi*t)
    u0 = lambda x: 0 * x
    u1 = lambda x: 2*np.pi*np.sin(2*np.pi*x)
    solve_wave_equ5(u0, u1, M=200, N=500, dt=0.002, interval=5, a=0.01, 
                    maxit=50, plot=True, filename='sinsinq5')
    
    # u = sin(2pi*x)cos(2pi*t)
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 0 * x
    solve_wave_equ5(u0, u1, M=200, N=500, dt=0.002, a=0.01, maxit=50, 
                    interval=5, plot=True, filename='sincosq5')
    
    # u = sin(2pi(x+t))
    u0 = lambda x: np.sin(2*np.pi*x)
    u1 = lambda x: 2*np.pi * np.cos(2*np.pi*x)
    solve_wave_equ5(u0, u1, M=200, N=500, dt=0.001, interval=5, a=0.1, 
                    maxit=50, plot=True, filename='sinq5')