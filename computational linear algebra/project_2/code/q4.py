import numpy as np
import cla_utils as cla
from scipy.sparse import csgraph
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt


def createA(m):
    '''
    Creates A  (mxm) as suggested in question 4 (A = I + L), for a random
    graph.

    Parameters
    ----------
    m : integer
        Determines the size of A (mxm) .

    Returns
    -------
    A : mxm numpy array
        A created as suggested iin Question 4.

    '''
    graph = np.random.randint(0,m,(m,m)) # Create random graph
    L = csgraph.laplacian(graph) # Laplacian of graph
    A = L + np.eye(m)
    return A


if __name__ == "__main__":
    
    # Find the c such that the preconditioned system M^-1Ax = M^-1b, where
    # M = cU has the smalest maximum (absolute) eigenvalue.
    for m in [100, 200, 300, 400, 500]:
        np.random.seed(m)
        A = createA(m)
        U = np.triu(A)
        maxeigs = np.zeros(25)
        # Through trial and error i find the appropriate c to be within this 
        # range.
        cs = np.linspace(0.5,0.515,25) 
        for i in range(25):
            M = cs[i]*U
            eig, _ = np.linalg.eig(solve_triangular(M, A))
            maxeig = max(abs(eig-1))
            maxeigs[i] = maxeig
        plt.plot(cs, maxeigs, label = m)
    plt.legend(title='Matrix size')
    plt.title(label = '$|\lambda -1|_{max}$ for various matrices while varying $c_1$')
    plt.xlabel('$c_1$')
    plt.ylabel('$|\lambda -1|_{max}$')
    plt.savefig('Figures/cvseig.png', dpi=600)
    plt.show()
   
    # Considering 500x500 matrices
    c = cs[np.argmin(maxeigs)]   
    print('For 500x500 matrices the optimal $c_1$ is', c)
    
    for i in range(5):
        np.random.seed(m)
        A = createA(m)
        # Forming preconditioned system
        U = np.triu(A)
        M = c*U
        apply_pc = lambda x: solve_triangular(M, x)
        x0 = np.random.randint(0, m, m)
        b = A@x0
        b0 = np.array(b)
        
        
        x, nits, r = cla.GMRES(A, b, maxit=100, tol=10**-8, 
                                     return_residual_norms=True)
        if max(abs(x - x0)) > 10**-3:
            print('Non-preconditioned system doesnt return a valid solution')
        if nits == -1:
            print('Iteration:%d Non-preconditioned system doesnt converge'
                  % (i+1))
        
        
        xpc, nitspc, rpc = cla.GMRES(A, b, maxit=100, tol= 10**-8, 
                                     return_residual_norms=True, 
                                     apply_pc=apply_pc)
        if max(abs(x - x0)) > 10**-3:
            print('Preconditioned system doesnt return a valid solution')  
        if nitspc == -1:
            print('Iteration: %d Preconditioned system doesnt converge'
                  % (i+1))
    
    
    # Eigenvalues of preconditioned and non preconditioned system    
    evals = np.linalg.eigvals(A)
    maxevals = max(abs(evals))
    minevals = min(abs(evals))
    print('Max abs eigenvalue = %.2f, min abs eigenvalue = %.2f without pc.'
          % (maxevals, minevals))
    evalspc = np.linalg.eigvals(apply_pc(A))
    maxevalspc = max(abs(evalspc))
    minevalspc = min(abs(evalspc))
    print('Max abs eigenvalue = %.2f, min abs eigenvalue = %.2f with pc.'
          % (maxevalspc, minevalspc))
    
    
    c = max(abs(evalspc-1)) # |1-lambda|<=c
    print('c=', c)
    cs =  np.repeat(c, nitspc) ** np.arange(1, nitspc+1)
    cs *= np.linalg.norm(apply_pc(b))
        
    plt.plot(np.arange(1, nitspc+1), cs, label='$c^n|M^{-1}b|$')
    plt.plot(np.arange(1, nitspc+1), rpc, label='$|M^{-1}Ax - M^{-1}b|$')
    plt.title(label='Upperbound and residual against iterations')
    plt.yscale('log')
    plt.ylabel('Log errors')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig('Figures/4d2.png', dpi=600)
    plt.show()
    
    