from xsect_majorana_tch import xsect, xsectk2_part_wave
import numpy as np
from scipy.special import roots_laguerre
from xsect_sch import xsect_sch, xsect_intf, xsect_intf_part_wave
from phase_shift import phase_shift

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
    integ = sum(integ_arr)

    if maxn > minn:
        # add sampling points until convergence
        integ_err = 1.0

        for n in np.arange(minn+1, maxn+1,1):
            nodes, weights = roots_laguerre(n)
            
            integ_old = integ
            integ_arr = list(map(lambda i: np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect(mX, mphi, alphaX, np.sqrt(2*nodes[i])*v0, sign, *params), range(n)))
            
            integ = sum(integ_arr)
            integ_err = np.abs(integ-integ_old)/integ_old

            if integ_err < precision:
                break
     
    return integ

"""
Velocity averaged cross section for s-ch resonance
"""


def mean_xsectv_sch(mX, mphi, alphaX, vmean, width_sm, n_width=100, eval_N_peak=500, tol_peak=1e-3, h=1e-1, eval_N=5, tol_trp=1e-2):
    """
    Return s-channel <\sigma v>, assuming Boltzmann distribution.
    
    Note: the input velocity is vmean, <v> = 2\sqrt(2/pi)v0, not v0.

    n_width: if the resonance exists, we integrate around the peak with interval h=(half width)/n_width
    eval_N_peak: for iteration > eval_N_peak, we evaluate convergence
    tol_peak: tolerance for the integration around the resonance

    h: the integration interval outside the resonance
    eval_N: for iteration > eval_N, we evaluate convergence
    tol_trp: tolerance of this region
    """
    
    v0 = np.sqrt(np.pi/8)*vmean
    vRsq = mphi**2/mX**2 - 4

    def integration_outside_peak(h):
        # Integrate outside the resonance peak
        # We do not care the overlap on peak by taking the integration interval
        # larger than max_peak_width
        if vRsq >= 0 and h < max_peak_width:
            h = max_peak_width

        # Peak of x^3exp(-x^2)
        xGauss_peak = np.sqrt(3.0/2.0)

        # Points at which x^3exp(-x^2) beceoms 10^-3 smaller than the peak value
        xi = 0.074428
        xf = 3.3849

        # integration for x < xGauss_peak
        xs = np.arange(xGauss_peak, xi, -h)
        integ_arr_low = np.array([])
        for i, x in enumerate(xs):
            integ_arr_low = np.append(integ_arr_low, h*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_sch(mX, mphi, alphaX, np.sqrt(2)*v0*x, width_sm))

            if i>eval_N:
                eval_err = np.sum(integ_arr_low[-eval_N:])/np.sum(integ_arr_low[:-eval_N])
                if eval_err < tol_trp: break

        # integraion for x > xGauss_peak
        xs = np.arange(xGauss_peak+h, xf, h)
        integ_arr_high = np.array([])
        for i, x in enumerate(xs):
            integ_arr_high = np.append(integ_arr_high, h*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_sch(mX, mphi, alphaX, np.sqrt(2)*v0*x, width_sm))

            if i>eval_N:
                eval_err = np.sum(integ_arr_high[-eval_N:])/np.sum(integ_arr_high[:-eval_N])
                if eval_err < tol_trp: break

        return np.sum(integ_arr_low) + np.sum(integ_arr_high)

    if vRsq >= 0:
        # Integraion with mediator resonance
        vR = np.sqrt(vRsq)
        wR = alphaX*vR**3/8 + mphi*width_sm/mX**2 # width around peak

        # We first integrate the narrow region around the peak
        # we use x, v=sqrt(2)v0*x
        xR = vR/np.sqrt(2)/v0 
        width_xR = np.sqrt(2.)*wR/4.0/vR/v0 # half width around xR
    
        max_peak_width = width_xR*10 # We integrate around the peak at most [xR-max_peak_width, xR+max_peak_width]
        h_peak = width_xR/n_width # intagration interval around the peak
        
        integ_arr = np.array([h_peak*4*np.sqrt(2/np.pi)*v0*xR**3*np.exp(-xR**2)*xsect_sch(mX, mphi, alphaX, np.sqrt(2)*v0*xR, width_sm)])
        xs = np.arange(xR+h_peak, xR+max_peak_width, h_peak)
        for i, x in enumerate(xs):
            integ_arr = np.append(integ_arr, h_peak*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_sch(mX, mphi, alphaX, np.sqrt(2)*v0*x, width_sm))
            x = xR - (i+1)*h_peak
            integ_arr = np.append(integ_arr, h_peak*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_sch(mX, mphi, alphaX, np.sqrt(2)*v0*x, width_sm))
            if i>eval_N_peak:
                eval_err = np.sum(integ_arr[-eval_N:])/np.sum(integ_arr[:-eval_N])
                if eval_err < tol_peak: break

        return np.sum(integ_arr) + integration_outside_peak(h)

    else:
        return integration_outside_peak(h)


def mean_xsectv_intf(mX, mphi, alphaX, vmean, sign, width_sm,
                     n_width=100, eval_N_peak=500, tol_peak=1e-3, h=1e-1, eval_N=5, tol_trp=1e-2,
                     minn=10, maxn=15, tol_quad=1e-2,
                     delta_arr_input=None, minb=0.1, 
                     tol=1e-2, eval_L=2, max_L=260,
                     method='RK45', tol_convergence=0.01, max_itr=100,
                     xtol_root=1e-4, rtol_root=1e-4,
                     atol_ode=1e-4, rtol_ode=1e-5):
    
    """
    Return <\sigma v>, assuming Boltzmann distribution.
    
    Note: the input velocity is vmean, <v> = 2\sqrt(2/pi)v0, not v0.

    n_width: if the resonance exists, we integrate around the peak with intercal h=(half width)/n_width
    eval_N_peak: for iteration > eval_N_peak, we evaluate convergence
    tol_peak: tolerance for the integration around the resonance

    h: the integration interval outside the resonance
    eval_N: for iteration > eval_N, we evaluate convergence
    tol_trp: tolerance of this region
    """
    other_params = (delta_arr_input, minb,
                    tol, eval_L,  max_L,
                    method, tol_convergence, max_itr,
                    xtol_root, rtol_root,
                    atol_ode, rtol_ode)
    
    v0 = np.sqrt(np.pi/8)*vmean
    vRsq = mphi**2/mX**2 - 4

    if vRsq >= 0:
        # Integraion with mediator resonance
        vR = np.sqrt(vRsq)
        wR = alphaX*vR**3/8 + mphi*width_sm/mX**2 # width around peak

        # We first integrate the narrow region around the peak
        # we use x, v=sqrt(2)v0*x
        xR = vR/np.sqrt(2)/v0 
        width_xR = np.sqrt(2.)*wR/4.0/vR/v0 # half width around xR
    
        max_peak_width = (wR/2.0/v0**2)**2/2.0/xR / 5.0 # We integrate around the peak at most [xR-max_peak_width, xR+max_peak_width]
        h_peak = width_xR/n_width # intagration interval around the peak
        
        integ_arr = np.array([h_peak*4*np.sqrt(2/np.pi)*v0*xR**3*np.exp(-xR**2)*xsect_intf(mX, mphi, alphaX, np.sqrt(2)*v0*xR, sign, width_sm, *other_params)])
        xs = np.arange(xR+h_peak, xR+max_peak_width, h_peak)
        for i, x in enumerate(xs):
            integ_arr = np.append(integ_arr, h_peak*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_intf(mX, mphi, alphaX, np.sqrt(2)*v0*x, sign, width_sm, *other_params))
            x = xR - (i+1)*h_peak
            integ_arr = np.append(integ_arr, h_peak*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_intf(mX, mphi, alphaX, np.sqrt(2)*v0*x, sign, width_sm, *other_params))
            if i>eval_N_peak:
                eval_err = np.sum(integ_arr[-eval_N:])/np.sum(integ_arr[:-eval_N])
                if eval_err < tol_peak: break

        # Then we integrate outside the peak
        # We do not care the overlap on peak by taking the integration interval
        # larger than max_peak_width
        
        if h < max_peak_width:
            h = max_peak_width

        # Peak of x^3exp(-x^2)
        xGauss_peak = np.sqrt(3.0/2.0)

        # Points at which x^3exp(-x^2) beceoms 10^-3 smaller than the peak value
        xi = 0.074428
        xf = 3.3849

        # integration for x < xGauss_peak
        xs = np.arange(xGauss_peak, xi, -h)
        integ_arr_low = np.array([])
        for i, x in enumerate(xs):
            integ_arr_low = np.append(integ_arr_low, h*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_intf(mX, mphi, alphaX, np.sqrt(2)*v0*x, sign, width_sm, *other_params))

            if i>eval_N:
                eval_err = np.sum(integ_arr_low[-eval_N:])/np.sum(integ_arr_low[:-eval_N])
                if eval_err < tol_trp: break

        # integraion for x > xGauss_peak
        xs = np.arange(xGauss_peak+h, xf, h)
        integ_arr_high = np.array([])
        for i, x in enumerate(xs):
            integ_arr_high = np.append(integ_arr_high, h*4*np.sqrt(2/np.pi)*v0*x**3*np.exp(-x**2)*xsect_intf(mX, mphi, alphaX, np.sqrt(2)*v0*x, sign, width_sm, *other_params))

            if i>eval_N:
                eval_err = np.sum(integ_arr_high[-eval_N:])/np.sum(integ_arr_high[:-eval_N])
                if eval_err < tol_trp: break
                
        return np.sum(integ_arr) + np.sum(integ_arr_low) + np.sum(integ_arr_high)
    
    else:
        # integraion without resonance
        
        # First minn sampling
        nodes, weights = roots_laguerre(minn)
        integ_arr = list(map(lambda i: np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect_intf(mX, mphi, alphaX, np.sqrt(2*nodes[i])*v0, sign, width_sm, *other_params), range(minn)))
        integ = sum(integ_arr)

        if maxn > minn:
            # add sampling points until convergence
            integ_err = 1.0

            for n in np.arange(minn+1, maxn+1,1):
                nodes, weights = roots_laguerre(n)
                
                integ_old = integ
                integ_arr = list(map(lambda i: np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect_intf(mX, mphi, alphaX, np.sqrt(2*nodes[i])*v0, sign, width_sm, *other_params), range(minn)))
                
                integ = sum(integ_arr)
                integ_err = np.abs(integ-integ_old)/integ_old

                if integ_err < tol_quad:
                    break

        return integ


def mean_xsectv_tu_intf(mX, mphi, alphaX, vmean, sign, width_sm,
                        minn=10, maxn=15,
                        precision=1e-2,
                        minb=0.1, classical_th = 10,
                        tol=1e-2, eval_L=2, max_L=260,
                        method='RK45', tol_convergence=0.01, max_itr=100,
                        xtol_root=1e-4, rtol_root=1e-4,
                        atol_ode=1e-4, rtol_ode=1e-5,
                        n_width=100, eval_N_peak=500, tol_peak=1e-3, h=1e-1, eval_N=5, tol_trp=1e-2, tol_quad=1e-2):
    """
    Calculate t+u corss section and interference btw s- and t+u channel simultaneously
    For the t-channel resonance region, this function reduces the computation time.
    """

    v0 = np.sqrt(np.pi/8)*vmean
    vRsq = mphi**2/mX**2 - 4

    b = alphaX*mX/mphi

    # If s-ch resonance exists or in Born region
    if vRsq > 0 or b < minb:
        #t+u-channel
        params = (mX, mphi, alphaX, vmean, sign, minn, maxn,
                  precision,
                  minb, classical_th,
                  tol, eval_L, max_L,
                  method, tol_convergence, max_itr,
                  xtol_root, rtol_root,
                  atol_ode, rtol_ode)
        xsectv_t = mean_xsectv(*params)

        #interference btw. s- and t-(u-) channel
        params=(mX, mphi, alphaX, vmean, sign, width_sm,
                n_width, eval_N_peak, tol_peak, h, eval_N, tol_trp,
                minn, maxn, tol_quad,
                None, minb,
                tol, eval_L, max_L,
                method, tol_convergence, max_itr,
                xtol_root, rtol_root,
                atol_ode, rtol_ode)
        xsectv_intf = mean_xsectv_intf(*params)

        return xsectv_t + xsectv_intf

    # In t-ch resonance region
    else:
        other_params = (tol, eval_L,  max_L,
                        method, tol_convergence, max_itr,
                        xtol_root, rtol_root,
                        atol_ode, rtol_ode)
        integ_err = 1.0
        integ = 1.0
        
        for n in np.arange(minn, maxn+1, 1):
            integ_old = integ
            
            nodes, weights = roots_laguerre(n)
            integ_t_arr = np.array([])
            integ_intf_arr = np.array([])

            # For each nodes of Quadrature
            for i in range(n):
                v = np.sqrt(2*nodes[i]) * v0
                a = v/(2*alphaX)

                if mX*v/mphi < classical_th:
                    # t+u channel
                    xsectk2_t = xsectk2_part_wave(a, b, sign, *other_params)
                    xsect_t = xsectk2_t[0]*16*np.pi/v**2/mX**2
                    delta_arr_input = xsectk2_t[-1]
                    #print delta_arr_input
                    
                    integ_t_arr = np.append(integ_t_arr, np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect_t)
                    
                    # Interference term
                    xsect_intf = xsect_intf_part_wave(mX, mphi, alphaX, v, sign, width_sm, delta_arr_input, *other_params)[0]
                    integ_intf_arr = np.append(integ_intf_arr, np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect_intf)
                else:
                    xsect_t = xsect_clas(mX, mphi, alphaX, v, sign)
                    integ_t_arr = np.append(integ_t_arr, np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect_t)

                    #interference btw. s- and t-(u-) channel
                    xsect_intf = xsect_intf_part_wave(mX, mphi, alphaX, v, sign, width_sm, None, *other_params)[0]
                    integ_intf_arr = np.append(integ_intf_arr, np.sqrt(8/np.pi)*v0*weights[i]*nodes[i]*xsect_intf)
            
            integ = np.sum(integ_t_arr + integ_intf_arr)

            if maxn > minn:
                # add sampling points until convergence
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
    width_sm = 0.0

    #print(mean_xsectv(mX, mphi, alphaX, vmean, sign))
    print(mean_xsectv_sch(mX, mphi, alphaX, vmean, width_sm))
    print(mean_xsectv_intf(mX, mphi, alphaX, vmean, sign, width_sm))
    print(mean_xsectv_tu_intf(mX, mphi, alphaX, vmean, sign, width_sm))
