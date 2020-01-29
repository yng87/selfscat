from xsect_majorana_tch import xsect
import numpy as np
from scipy.special import roots_laguerre

"""
<\sigma v>, velocity averaged cross section times relative velocity,
assuming Maxwell-Boltzmann distribution.
"""

def mean_xsectv(mX, mphi, alphaX, vmean, sign, minn=10, maxn=15,
                precision=1e-2,
                minb=0.1, classical_th = 10, 
                tol=1e-2, eval_L=2, max_L=260,
                method='RK45', tol_convergence=0.01, max_itr=100,
                xtol_root=1e-4, rtol_root=1e-4,
                atol_ode=1e-4, rtol_ode=1e-5):
    """
    Return <\sigma v>, assuming Boltzmann distribution.
    Only t+u channel
    We use Gauss-Laguerre quadrature for integration.
    
    Note: the input velocity is vmean, <v> = 2\sqrt(2/pi)v0, not v0.
    minn: we sample at least minn points by quadrature.
    maxn: we do not sample more than maxn, if maxn <= minn, we do not check convergence.
    precision: we stop integration when it converges below this value
    Other parameters are for cross section
    """

    v0 = np.sqrt(np.pi/8)*vmean
    
    params = (minb, classical_th,
              tol, eval_L, max_L,
              method, tol_convergence, max_itr,
              xtol_root, rtol_root,
              atol_ode, rtol_ode)

    # First minn sampling
    nodes, weights = roots_laguerre(minn)
    integ_arr = list(map(lambda i: np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect(mX, mphi, alphaX, np.sqrt(2*nodes[i])*v0, sign, *params), range(minn)))
    integ = np.sum(integ_arr)

    if maxn > minn:
        # add sampling points until convergence
        integ_err = 1.0

        for n in np.arange(minn+1, maxn+1,1):
            nodes, weights = roots_laguerre(n)
            
            integ_old = integ
            integ_arr = list(map(lambda i: np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect(mX, mphi, alphaX, np.sqrt(2*nodes[i])*v0, sign, *params), range(n)))
            
            integ = np.sum(integ_arr)
            integ_err = np.abs(integ-integ_old)/integ_old

            if integ_err < precision:
                break
     
    return integ


if __name__ == "__main__":
    # execute only if run as a script
    mX = 10.
    mphi = 1e-2
    alphaX = 1e-3
    sign = 1
    vmean = 1e-3

    res = mean_xsectv(mX, mphi, alphaX, vmean, sign)

    print(res)

