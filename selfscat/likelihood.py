import numpy as np
from mean_xsectv_majorana import mean_xsectv, mean_xsectv_sch, mean_xsectv_intf, mean_xsectv_tu_intf

GeVinvTocm = 0.197e-13
GeVtogram = 1.78e-24
c = 3e5 #km/s

def likelihood(logxsect, delta_logxect, 
               mX, mphi, alphaX, vmean, sign, width_sm,
               minn=10, maxn=15,
               precision=1e-2,
               minb=0.1, classical_th=10,
               tol=1e-2, eval_L=2, max_L=260,
               method='RK45', tol_convergence=0.01, max_itr=100,
               xtol_root=1e-4, rtol_root=1e-4,
               atol_ode=1e-4, rtol_ode=1e-5,
               n_width=100, eval_N_peak=500, tol_peak=1e-3, h=1e-1, eval_N=5, tol_trp=1e-2, tol_quad=1e-2):
    """
    Calculate log likelihood.
    
    Note: the input is vmean, <v> = 2\sqrt(2/pi)v0.
    The logs here are all log10.
    Mass units should be inputed as [GeV]

    logxect: the observed DM cross section for a certain astrophysical object (log10)
    delta_logxect: the uncertainty of logxect
    Other parameters: see xsect.py
    """

    #s-channel
    params = (mX, mphi, alphaX, vmean, width_sm,
              n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp)
    xsectv_s = mean_xsectv_sch(*params)

    # (t+u)-channel + interference btw. s- and (t+u) channel
    params=(mX, mphi, alphaX, vmean, sign, width_sm,
            minn, maxn, precision,
            minb, classical_th,
            tol, eval_L, max_L,
            method, tol_convergence, max_itr,
            xtol_root, rtol_root,
            atol_ode, rtol_ode,
            n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp, tol_quad)
    xsectv_tu_intf = mean_xsectv_tu_intf(*params)
    
    log_mean_xsectv = np.log10( (xsectv_tu_intf+xsectv_s)* GeVinvTocm**2/(mX*GeVtogram)*c)

    chisq = (log_mean_xsectv - logxsect)**2/2.0/delta_logxect**2
    return np.exp(-chisq)/delta_logxect/np.sqrt(2*np.pi)
    #return chisq


def likelihood_HB_data(filepath,
                        mX, mphi, alphaX, sign, width_sm,
                        minn=10, maxn=15,
                        precision=1e-2,
                        minb=0.1, classical_th=10,
                        tol=1e-2, eval_L=2, max_L=260,
                        method='RK45', tol_convergence=1e-2, max_itr=100,
                        xtol_root=1e-4, rtol_root=1e-4,
                        atol_ode=1e-6, rtol_ode=1e-6,
                        n_width=100, eval_N_peak=500, tol_peak=1e-3, h=1e-1, eval_N=5, tol_trp=1e-2, tol_quad=1e-2):

    """
    Read data, and iterate likelihood calculation.
    """
    
    data = np.genfromtxt(filepath, names=('v', 'sigmav', 'verr', 'sigmaverr'), comments='#')

    res = []
    for d in data:
        logxsect = d['sigmav']
        delta_logxect = d['sigmaverr']
        vmean = 10**(d['v']) / c

        params = (mX, mphi, alphaX, vmean, sign, width_sm,
                  minn, maxn,
                  precision,
                  minb, classical_th,
                  tol, eval_L, max_L,
                  method, tol_convergence, max_itr,
                  xtol_root, rtol_root,
                  atol_ode, rtol_ode,
                  n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp, tol_quad)

        L = likelihood(logxsect, delta_logxect, *params)
        print(L)
        res.append(L)
        
    return res


"""
---------------------------------
Below is for internal cross check
---------------------------------
"""

def likelihood_separate(logxsect, delta_logxect, 
                        mX, mphi, alphaX, vmean, sign, width_sm,
                        minn=10, maxn=15,
                        precision=1e-2,
                        minb=0.1, classical_th=10,
                        tol=1e-2, eval_L=2, max_L=260,
                        method='RK45', tol_convergence=0.01, max_itr=100,
                        xtol_root=1e-4, rtol_root=1e-4,
                        atol_ode=1e-4, rtol_ode=1e-5,
                        n_width=100, eval_N_peak=500, tol_peak=1e-3, h=1e-1, eval_N=5, tol_trp=1e-2, tol_quad=1e-2):
    """
    Calculate likelihood.
    
    Note: the input is vmean, <v> = 2\sqrt(2/pi)v0.
    The logs here are all log10.
    Mass units should be inputed as [GeV]

    logxect: the observed DM cross section for a certain astrophysical object (log10)
    delta_logxect: the uncertainty of logxect
    Other parameters: see xsect.py
    """

    #s-channel
    params = (mX, mphi, alphaX, vmean, width_sm,
              n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp)
    xsectv_s = mean_xsectv_sch(*params)

    # t-channel
    params=(mX, mphi, alphaX, vmean, sign,
            minn, maxn, precision,
            minb, classical_th,
            tol, eval_L, max_L,
            method, tol_convergence, max_itr,
            xtol_root, rtol_root,
            atol_ode, rtol_ode)
    xsectv_tu = mean_xsectv(*params)

    # interference
    params=(mX, mphi, alphaX, vmean, sign, width_sm,
            n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp,
            minn, maxn, tol_quad,
            None, minb, 
            tol, eval_L, max_L,
            method, tol_convergence, max_itr,
            xtol_root, rtol_root,
            atol_ode, rtol_ode)
    xsectv_intf = mean_xsectv_intf(*params)
    
    log_mean_xsectv = np.log10((xsectv_tu+xsectv_s+xsectv_intf)* GeVinvTocm**2/(mX*GeVtogram)*c)

    chisq = (log_mean_xsectv - logxsect)**2/2.0/delta_logxect**2
    return np.exp(-chisq)/delta_logxect/np.sqrt(2*np.pi)
 
    
def test(filepath,
         mX, mphi, alphaX, sign, width_sm,
         minn=10, maxn=15,
         precision=1e-2,
         minb=0.1, classical_th=10,
         tol=1e-2, eval_L=2, max_L=260,
         method='RK45', tol_convergence=1e-2, max_itr=100,
         xtol_root=1e-4, rtol_root=1e-4,
         atol_ode=1e-6, rtol_ode=1e-6,
         n_width=100, eval_N_peak=500, tol_peak=1e-3, h=1e-1, eval_N=5, tol_trp=1e-2, tol_quad=1e-2):

    """
    Read data, and iterate likelihood calculation.
    """
    import time

    data = np.genfromtxt(filepath, names=('v', 'sigmav', 'verr', 'sigmaverr'), comments='#')

    #minn = 10
    #maxn = 100

    d = data[-6]
    logxsect = d['sigmav']
    delta_logxect = d['sigmaverr']
    vmean = 10**(d['v']) / c

    # for n in range(minn, maxn+1, 1):
    #     params = (mX, mphi, alphaX, vmean, sign, width_sm,
    #               n, maxn,
    #               precision,
    #               minb, classical_th,
    #               tol, eval_L, max_L,
    #               method, tol_convergence, max_itr,
    #               xtol_root, rtol_root,
    #               atol_ode, rtol_ode,
    #               n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp, tol_quad)
        
    #     start = time.time()
    #     L1 = likelihood(logxsect, delta_logxect, *params)
    #     end = time.time()
    #     t1 = end-start
    #     print 'n={}: L1 = {} ({:.3f} sec)'.format(n, L1, t1)
    print('-----------------------------------------')
    print('classical_th = ', classical_th)
    for d in data:
        logxsect = d['sigmav']
        delta_logxect = d['sigmaverr']
        vmean = 10**(d['v']) / c

        params = (mX, mphi, alphaX, vmean, sign, width_sm,
                  minn, maxn,
                  precision,
                  minb, classical_th,
                  tol, eval_L, max_L,
                  method, tol_convergence, max_itr,
                  xtol_root, rtol_root,
                  atol_ode, rtol_ode,
                  n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp, tol_quad)

        start = time.time()
        L1 = likelihood(logxsect, delta_logxect, *params)
        end = time.time()
        t1 = end-start

        # start = time.time()
        # L2= likelihood_separate(logxsect, delta_logxect, *params)
        # end = time.time()
        # t2 = end-start

        print('L1 = {:.3f} ({:.3f} sec)'.format(L1, t1))

    # print '-----------------------------------------'
    # print 'classical_th = ', classical_th
    # for d in data:
    #     logxsect = d['sigmav']
    #     delta_logxect = d['sigmaverr']
    #     vmean = 10**(d['v']) / c

    #     params = (mX, mphi, alphaX, vmean, sign, width_sm,
    #               minn, maxn,
    #               precision,
    #               minb, classical_th,
    #               tol, eval_L, max_L,
    #               method, tol_convergence, max_itr,
    #               xtol_root, rtol_root,
    #               atol_ode, rtol_ode,
    #               n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp, tol_quad)

    #     start = time.time()
    #     L1 = likelihood(logxsect, delta_logxect, *params)
    #     end = time.time()
    #     t1 = end-start

    #     start = time.time()
    #     L2= likelihood_separate(logxsect, delta_logxect, *params)
    #     end = time.time()
    #     t2 = end-start

    #     print 'L1 = {:.3f} ({:.3f} sec), L2 = {:.3f} ({:.3f} sec)'.format(L1, t1, L2, t2)


def main():
    filepath = '../sample_data/fitted_xsect_log.dat'

    mX = 10
    mphi = 0.01
    alphaX = 0.00099
    sign = 1.0
    minn = 10
    maxn = 12
    width_sm = 0.0

    params = (mX, mphi, alphaX, sign, width_sm,  minn, maxn)

    #likelihood_fit_data(filepath, *params)
    test(filepath, *params)

if __name__ == '__main__':
    main()
    
    
