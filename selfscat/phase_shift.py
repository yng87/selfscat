import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

'''
Phase shift calculation following Tulin, Yu and Zurek (2013) [arXiv:1302.3898].

Throughout this function, the following parameters are used:

a = v/(2*alphaX), where v is in natural unt (dimensionless).
b = mX*alphaX/mphi
l : partial wave (0 for s-wave, 1 for p-wave, ...)
sign: +1 for attractive force. -1 for repulsive force
'''

def phase_shift_fixed_range(a, b, l, sign, xi, xf,
                            method='RK45', atol_ode=1e-4, rtol_ode=1e-5):
    '''
    Calculate phase shift of Yukawa potential.
    This function calculates the phase shift for fixed range [xi, xf].

    Inputs:
    x = alphaX*mX*r.
    atol_ode and rtol_ode is precision paramters for ODE solver.

    Outputs:
    dl : phase shift
    '''

    def f(x, y):
        return [y[1], -(a**2 - l*(l+1.0)/x**2 +sign*np.exp(-x/b)/x)*y[0]]

    def jac(x, y):
        return [[0.0, 1.0], [-(a**2 - l*(l+1.0)/x**2 +sign*np.exp(-x/b)/x), 0.0]]

    # solve ODE
    t_eval=None
    y0 = [1.0, (l+1)/xi]

    sol = solve_ivp(fun=f,
                    t_span=(xi, xf),
                    t_eval=t_eval,
                    y0 = y0,
                    jac=jac,
                    method=method,
                    atol=atol_ode,
                    rtol=rtol_ode)

    # calculate phase shift
    bl = xf*sol.y[1][-1]/sol.y[0][-1] - 1.0
    tandl = (a*xf*spherical_jn(l,a*xf,derivative=True) - bl*spherical_jn(l,a*xf, derivative=False))\
            /(a*xf*spherical_yn(l,a*xf,derivative=True) - bl*spherical_yn(l,a*xf, derivative=False))
    dl = np.arctan(tandl)

    return dl


def phase_shift(a, b, l, sign,
                   method='RK45',
                   tol_convergence=0.01, max_itr=100,
                   xtol_root=1e-4, rtol_root=1e-4,
                   atol_ode=1e-4, rtol_ode=1e-5):

    '''
    This function iterates the above phase_shift_fixed_range until convergence.

    tol_convergence, max_itr, xtol_root, ... are hyper parameters,
    which is optimized for l ~ 100.
    '''

    # Internal parameters
    largei = 1000.0
    largef = 3e4


    # Initial xi
    xi = np.min([b, (l+1.0)/a])/largei

    # Initial xf. This part is tricky, and I suspect there is a better way...,
    xfmin = brentq(lambda x: x*a**2/100 - np.exp(-x/b), 0.0, 1e20,
                    xtol=xtol_root, rtol=rtol_root)
    xfmax = brentq(lambda x: x*a**2/largef - np.exp(-x/b), 0.0, 1e20, 
                    xtol=xtol_root, rtol=rtol_root)

    if xfmin <= xi: xfmin = 2*xi
    if a*xfmax < l+1: xfmax = float(l+1)/a
    if xfmax <= xfmin: xfmax = 10*xfmin

    # Loop for convergence
    for i in range(max_itr):
        xfs = np.logspace(np.log10(xfmin), np.log10(xfmax), 1000*(2*l+1)*(i+1))
        dl = 100.0
        conv = 1.0

        # Loop with increasing xf:
        for xf in xfs:
            dl_old = dl
            dl = phase_shift_fixed_range(a, b, l, sign, xi, xf, method, atol_ode, rtol_ode)

            # I found if xi is too small, calculation fails.
            if np.isnan(dl):
                xi *= 2 #(1+tol_convergence/(l+1.0))
                dl = dl_old
                continue

            # Check convergence
            conv = np.abs(dl/dl_old - 1)

            # I also check whether the potential part is sufficiently small or not
            if conv < tol_convergence and (np.exp(-xf/b)/xf)/a**2 < tol_convergence:
                success = True
                break
            # If the convergence is not enough, decrease xi
            else:
                success = False
                xi = xi / (1+tol_convergence/(l+1.0))
        if success: break

    return dl
