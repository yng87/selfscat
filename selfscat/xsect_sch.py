import numpy as np
from xsect_majorana_tch import legendre_integ
from phase_shift import phase_shift
"""
s-channel contribution
"""

def xsect_sch(mX, mphi, alphaX, v, width_sm):
    """
    S-channel contribution to tranfer cross section in unit of mass.
    Born limit formula
    (e.g., if mX and mphi are given by [GeV], xsect is [GeV^-2]

    mX: DM mass
    mphi: mediator mass
    alpha_X: yukawa coupling
    v: relative velocity in unit of c
    """
    
    vRsq = -4 + mphi**2/mX**2
    w = alphaX*v**3/8 + mphi*width_sm/mX**2
    return np.pi*alphaX**2 / (16*mX**2) * v**4 / ((v**2 - vRsq)**2 + w**2)



"""
Interference between s- and t+u channel.
Usually this contribution is much smaller than the others,
so one may ignore it.
"""

def xsect_intf_born(mX, mphi, alphaX, v, width_sm):
    """
    Interference between s and t,u-channel using Born limit.
    Valid for b = alphaX*mX/mphi << 1.
    We use the non-relativistic expansion: epsilon = k^2/mX^2
    """

    epsilon = v**2/4
    r = mphi/mX
    vRsq = r**2 - 4
    w = alphaX*v**3/8 + mphi*width_sm/mX**2
    
    return -epsilon**2 * np.pi * alphaX**2 / mX**2 * (v**2-vRsq) / ((v**2-vRsq)**2 + w**2) * (2-3*r**2)/(3*r**4)


def xsect_intf_each_l(mX, mphi, alphaX, v, L, dl, width_sm):
    """
    Interference between s and t,u-channel for each partial wave L.
    """
    
    vRsq = -4 + mphi**2/mX**2
    epsilon = v**2 / 4
    w = alphaX*v**3/8 + mphi*width_sm/mX**2
    
    common_factor = np.pi*alphaX/mX**2 * v * (2*L+1) * np.sin(dl) * (np.cos(dl)*(v**2-vRsq) - np.sin(dl)*w)/ ((v**2 - vRsq)**2 + w**2)
    
    if L % 2 == 0:
        Legendre_factor = epsilon * (legendre_integ(0,L) - legendre_integ(1,L))
    else:
        Legendre_factor = -(1+epsilon/2) * 2 * (legendre_integ(1,L) -2*legendre_integ(2,L)/3 -legendre_integ(0,L)/3)

    return common_factor * Legendre_factor

def xsect_intf_part_wave(mX, mphi, alphaX, v, sign, width_sm, delta_arr_input=None,
                         tol=1e-2, eval_L=2, max_L=260,
                         method='RK45', tol_convergence=0.01, max_itr=100,
                         xtol_root=1e-4, rtol_root=1e-4,
                         atol_ode=1e-4, rtol_ode=1e-5):
    """
    Sum the interference terms over the partial wave L.
    To reduce the computational time, we make use of the phase shift already calculated by t,u-channel part.
    """
    a = v/(2*alphaX)
    b = alphaX*mX/mphi
    
    # params passed to phase shift calculater
    other_params = (method, tol_convergence, max_itr, xtol_root, rtol_root, atol_ode, rtol_ode)
    
    xsect_err = 1.0 # for error estimation

    if delta_arr_input == None:
        L0 = 0
        delta_arr = []
        xsect_arr = []
    else:
        delta_arr = delta_arr_input
        L0 = np.size(delta_arr)
        xsect_arr = list(map(lambda L:xsect_intf_each_l(mX, mphi, alphaX, v, L, delta_arr[L], width_sm), np.arange(L0)))

    for L in range(L0, max_L+1, 1):
        dl = phase_shift(a, b, L, sign, *other_params)
        delta_arr.append(dl)
        xsect_arr.append(xsect_intf_each_l(mX, mphi, alphaX, v, L, delta_arr[L], width_sm))
                
        if L >= eval_L:
            xsect = sum(xsect_arr[0:-eval_L])
            xsect_del = sum(xsect_arr[-eval_L:])
            xsect_err = np.abs(xsect_del/xsect)
            if xsect_err < tol and delta_arr[-1] < tol:
                break
            
    xsect = sum(xsect_arr)
    
    return xsect, xsect_arr

def xsect_intf(mX, mphi, alphaX, v, sign, width_sm, delta_arr_input=None,
               minb=1e-2,
               tol=1e-2, eval_L=2, max_L=260,
               method='RK45', tol_convergence=0.01, max_itr=100,
               xtol_root=1e-4, rtol_root=1e-4,
               atol_ode=1e-4, rtol_ode=1e-5):
    """
    Interference term.
    For b < minb, we use the Born formula
    """
    b = alphaX*mX/mphi

    if b < minb:
        return xsect_intf_born(mX, mphi, alphaX, v, width_sm)
    else:
        other_params = (tol, eval_L,  max_L,
                        method, tol_convergence, max_itr,
                        xtol_root, rtol_root,
                        atol_ode, rtol_ode)
        return xsect_intf_part_wave(mX, mphi, alphaX, v, sign, width_sm, delta_arr_input, *other_params)[0]


