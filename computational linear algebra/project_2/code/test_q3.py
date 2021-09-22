'''Tests for the second coursework Question 3'''
import pytest
import numpy as np
import q3


CID = 1357062


@pytest.mark.parametrize('m', [5, 10, 50, 100, 501, 1000])
def test_qr_factor_tri(m):
    np.random.seed(CID+m)
    x = np.random.rand(2*m -1)
    A = np.diag(x[:m]) + np.diag(x[m:],k=1) + np.diag(x[m:],k=-1)
    x, R = q3.qr_factor_tri(A, copy=True)
    # R is upper triagonal and has upper bandwidth 2
    assert(np.allclose(np.tril(np.triu(R),k=2) , R))
    Q = np.eye(m)
    for i in range(m-1):
        v = x[:, i]
        v.shape = (2, 1)
        Q[:i+2, i:i+2] += -2*(Q[:i+2, i:i+2]@v) @ v.T
    assert(np.allclose(np.eye(m), Q@Q.T)) # Check Q is orthogonal
    assert(np.allclose(np.triu(Q, k=-1), Q))# Check Q is upper Hessenberg
    
    assert(np.allclose(Q@R, A)) # Check A = QR
    
    RQ = R@Q
    assert(np.allclose(RQ, RQ.T)) # Check RQ is symmetric
    assert(np.allclose(np.triu(RQ, k=-1), RQ)) # Check RQ is upper Hessenberg


@pytest.mark.parametrize('m', [5, 10, 50, 100, 501, 1000])
def test_qr_alg_tri(m):
    np.random.seed(CID+m)
    x = np.random.rand(2*m -1)
    A = np.diag(x[:m]) + np.diag(x[m:],k=1) + np.diag(x[m:],k=-1)
    A0, _ = q3.qr_alg_tri(A, copy=True)
    assert(A0[-1,-2]<10**-12)
    assert(np.allclose(np.trace(A0), np.trace(A))) # Check trace is preserved
   
 
@pytest.mark.parametrize('m', [5, 10, 23, 25, 30])
def text_eigvalues_unshifted(m):
    np.random.seed(CID+m)
    x = np.random.rand(2*m -1)
    A = np.diag(x[:m]) + np.diag(x[m:],k=1) + np.diag(x[m:],k=-1)
    theirevals = np.sort(np.linalg.eigvalsh(A))
    A, _ = q3.q3e(A, copy=False, shift=False)
    myevals = np.sort(np.diag(A))
    errevals = myevals - theirevals
    assert(max(abs(errevals))<10**-3)

@pytest.mark.parametrize('m', [5, 10, 23, 25, 30])
def text_eigvalues_shifted(m):
    np.random.seed(CID+m)
    x = np.random.rand(2*m -1)
    A = np.diag(x[:m]) + np.diag(x[m:],k=1) + np.diag(x[m:],k=-1)
    theirevals = np.sort(np.linalg.eigvalsh(A))
    A, _ = q3.q3e(A, copy=False, shift=True)
    myevals = np.sort(np.diag(A))
    errevals = myevals - theirevals
    assert(max(abs(errevals))<10**-3)
