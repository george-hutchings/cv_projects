import numpy as np
import cla_utils as cla
import matplotlib.pyplot as plt


def sign(x):
    """
    Given a real number x, find the sign of the number, where sign(0) = 0.

    :param x: real number

    :return R: -1 or 1 depending on the sign of x
    """
    
    if x >= 0 :
        return 1
    else :
        return -1

def qr_factor_tri(A, copy=False):
    '''
    QR factorises A, a Tridiagonal symmetric matrix, returning a list of the 
    householder reflectors used instead of Q.

    Parameters
    ----------
    A : Tridiagonal symmetric mxm numpy array
        Matrix to be Q R factorised.
    copy : bool, optional
        Deterimnes whether the algorithm should be performed in-place, if False 
        algorithm will be performed in-place
        The default is False.

    Returns
    -------
    vs : 2x(m-1) numpy array
        Each column is the householder reflector used to QR factorise A, for 
        that respective iteration of the householder method.
    A : Upper triangular mxm numpy array
        R from the QR factorsation of A.

    '''
    if copy:
        A = np.array(A)
        
    if np.issubdtype(A[0,0], int):
        A = A.astype(float, copy=False)
        
    _, m = A.shape
    vs = np.zeros((2,m-1), dtype=A.dtype)
    for k in range(m-1):
        x = np.array(A[k:k+2, k])
        x[0] += sign(x[0]) * np.linalg.norm(x)
        x = np.divide(x, np.linalg.norm(x))
        vs[:, k] = x
        x = x[:, np.newaxis]
        A[k:k+2, k:k+3] += -2*x @ (x.T@A[k:k+2, k:k+3])
    return vs, A


def qr_alg_tri(A, copy=False):
    '''
    Performs the QR algorithm on the matrix A (mxm) which is of tridiagonal 
    symmetric form repeatin until the m,m-1 entry is <= 10^-12

    Parameters
    ----------
    A : Tridiagonal symmetric mxm numpy array
        Matrix for the QR algorthm to be applied.
    copy : bool, optional
        Deterimnes whether the algorithm should be performed in-place, if False 
        algorithm will be performed in-place
        The default is False.

    Returns
    -------
    T : mxm numpy array
        A after the application of the QR algorithm, with termination when 
        m,m-1 entry is <= 10^-12.
    err : list
        The absolute value of the m,m-1 entry of A after each iteration.

    '''
    T = A.astype('float64', copy=False)
    _, m = T.shape
    # Performing as list concatenation as opposed to numpy array concatenation
    # for efficiency.
    err = [abs(T[-1, -2])]
    while err[-1] >= 10**-12:
        vs , T = qr_factor_tri(T, copy=copy)
        for i in range(m-1):     
            v = vs[:, i]
            v = v[:, np.newaxis]
            T[:i+2, i:i+2] += -2*(T[:i+2, i:i+2]@v) @ v.T
        err.append(abs(T[-1, -2]))
    return T, err

def wilk_shift(A):
    '''
    Calculates the value of the wilkinson shift for the matrix A.

    Parameters
    ----------
    A : mxm numpy array
        Matrix for which the wilkinson shift is required.

    Returns
    -------
    mu : float
        Wilkinson shift for the matrix A.

    '''
    a = A[-1, -1]
    delta = (A[-2, -2]-a) * 0.5
    b = A[-1, -2]
    b2 = b**2
    mu = a - sign(delta)*b2/(abs(delta) + np.sqrt(delta**2 + b2))
    return mu
    



def q3e(A, copy=False, shift=False):
    '''
    Given a symmetric matrix A, finds its hessenberg form and then performs
    the QR algorithm on it such until its subdiagonal is <= 10^-12, this is 
    done by taking submatricies as explained in Question 3e.

    Parameters
    ----------
    A : Symmetric mxm numpy array
        Matrix to for the QR algorthm to be applied.
    copy : Bool, optional
        Deterimnes whether the algorithm should be performed in-place, if False 
        algorithm will be performed in-place
        The default is False.
    shift : Bool, optional
        Whether or not the Wilkinson shift should be applied. If False it will 
        not be applied The default is False.

    Returns
    -------
    A : mxm numpy array
        A_k, similar to A after k iterations of the QR algorithm.
    err : List 
        List containing the error after each iteration.

    '''
    A = cla.hessenberg(A, copy=copy)
    _, m = A.shape
    err = []
    for i in range(m-1):
        if abs(A[-1-i, -2-i]) >= 10**-12: # Avoid unneccessary calls to function
            if shift:
                muI = wilk_shift(A[:m-i, :m-i]) * np.eye(m-i)
                A[:m-i, :m-i] -= muI
            A[:m-i, :m-i], err0 = qr_alg_tri(A[:m-i, :m-i], copy=copy)
            if shift: A[:m-i, :m-i] += muI
            err = err + err0 # No need to iterate on 1x1 matrix
    return A, err


def plot3e(A, shift=False, markers=True, filename=None, pure_qr=False):
    '''
    Function to plot the iterations against the errors for q3e or pure_QR.

    Parameters
    ----------
    A : Symmetric mxm numpy array
        Matrix to for the QR algorthm to be applied.
    shift : Bool, optional
        Whether or not the Wilkinson shif should be applied. If False it will 
        not be applied The default is False.
    markers : Bool, optional
        Whether or not to include markers(points) on the plot, if True markers will be 
        included. The default is True.
    filename : String, optional
        The name to save the plot as. The default is None.
    pure_qr : Bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    if shift is True:
        colour = 'green'
    else:
        colour = 'blue'
    if markers is True:
        markers='o'
    else:
        markers=None
    
        
    if pure_qr is True:
        colour = 'orange'
        A, err = cla.pure_QR(A, 10000, 10**-12, errors=True)   
        x = np.arange(len(err)) + 1
    else:
        A, err = q3e(A, shift=shift)
        # Special arange, such that any increase doesnt count as an iteration
        x = np.ones(len(err), dtype=int)
        for i in range(1, len(err)):
            if err[i-1] < err[i]:
                x[i] = x[i-1]
            else:
                x[i] = x[i-1]+1    
    
    plt.plot(x, err, marker=markers, markersize=2, color=colour)
    plt.hlines(10**-12, xmin=1, xmax=x[-1], color='black', label='10^-12', linewidth = 0.8)
    plt.yscale('log')
    plt.legend()
    plt.title(label = 'Iteration vs error')
    plt.xlabel('Iteration')
    plt.ylabel('Log Error')
    plt.tight_layout()
    if filename is not None:
        plt.savefig('Figures/%s.png' % (filename),dpi=600)
    plt.show()
    return


if __name__ == "__main__":
    
    m=5
    A = np.outer(np.ones(m), np.arange(1,m+1)) + 1
    for i in range(1, m):
        A[i, :] =  A[i, :] + i 
    A = 1 / (A+1)
    plot3e(np.array(A), shift=False, markers=True, filename='Aij5unshifted')
    plot3e(np.array(A), shift=True, markers=True, filename='Aij5shifted')
    plot3e(np.array(A), pure_qr=True, markers=True, filename='Aij5pure_qr')
    
    theirevals = np.sort(np.linalg.eigvalsh(A))
    A, err = q3e(A, shift=False)
    myevals = np.sort(np.diag(A))
    errevals = myevals - theirevals
    print(theirevals)
    print(abs(errevals))

    
    np.random.seed(1234 * 30)
    A = np.random.rand(30,30)
    A = A + A.T
    plot3e(np.array(A), shift=False, markers=False, filename='rand30unshifted')
    plot3e(np.array(A), shift=True, markers=False, filename='rand30shifted')
    plot3e(np.array(A), pure_qr=True, markers=False, filename='rand30pure_qr')
    

    
    A = np.ones((15,15)) + np.diag(np.arange(14,-1,-1))
    plot3e(np.array(A), shift=False, markers=False, filename='15plus1unshifted')
    plot3e(np.array(A), shift=True, markers=False, filename='15plus1shifted')
    plot3e(np.array(A), pure_qr=True, markers=False, filename='15plus1pure_qr')
    
    theirevals = np.sort(np.linalg.eigvalsh(A))
    print(theirevals)
    
    
    
