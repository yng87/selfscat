from phase_shift import phase_shift
import numpy as np
from scipy.special import gammaln

'''
Functions for self-scattering cross section for Majorana fermion
interacting by Yukawa type potential.

Note that unlike Dirac fermion, there is interference between t- and u-channel.
'''


"""
small functions for summing partial waves
"""
def fmn(m, n):
    sign = (-1.0)**(((m+n+1)/2)%2)
    prefactor = sign /(m-n) /(m+n+1.0)

    if m < 5:
        mfactor = np.exp(gammaln(m+1)) / 2**m  / np.exp(2*gammaln(m/2.0+1))
    else:
        mfactor = np.sqrt(1.0/m)**3*(-1.0+4.0*m)/2/np.sqrt(2*np.pi)

    if n < 5:
        nfactor = np.exp(gammaln(n+1)) / 2**n *2.0 / np.exp(2*gammaln((n-1.0)/2.0+1))
    else:
        nfactor = np.sqrt(1.0/n)**3*(-1.0+8.0*n+32.0*n**2) / 16.0 / np.sqrt(2*np.pi)

    return prefactor * mfactor * nfactor


def legendre_integ(m, n):
    """
    \int_0^1 dx P_m(x)P_n(x), where P_m is Legendre polynomial.
    """
    if m >= 0 and  n >= 0 and m==n:
        return 1.0/(2.0*m + 1.0)
    elif m >= 0 and  n >= 0 and m%2==0 and n%2==1:
        return fmn(m,n)
    elif m >= 0 and  n >= 0 and m%2==1 and n%2==0:
        return fmn(n,m)
    else:
        return 0


def inside_sum(l1, l2, dl1, dl2):
    return np.exp(1.0j*dl1 - 1.0j*dl2)*np.sin(dl1)*np.sin(dl2)*((2.0*l1+1.0)*(2.0*l2+1.0)*legendre_integ(l1,l2) - (l1+1.0)*(2.0*l2+1.0)*legendre_integ(l1+1, l2) - l1*(2.0*l2+1.0)*legendre_integ(l1-1,l2))



"""
Transfer cross section of t+u channel by partial wave expansion
"""
def xsectk2_part_wave(a, b, sign, tol=1e-2, eval_L=2, max_L=260,
                      method='RK45', tol_convergence=0.01, max_itr=100,
                      xtol_root=1e-4, rtol_root=1e-4,
                      atol_ode=1e-4, rtol_ode=1e-5):
    """
    Return sigma_T*k^2/4pi, where k = mX*v/2

    Parameters are:
    a = v/(2*alphaX), where v is in natural unt (dimensionless).
    b = mX*alphaX/mphi
    sign: +1 for attractive force. -1 for repulsive force
    tol: partial wave summation stops when cross section and phase shift converges at this value
    eval_L: we sum at least eval_L partial waves (default: 2)
    max_L: we do not sum partial wave beyond this value

    Other parameters are related to phase shift calculation. See phase_shift.py
    """
    # params passed to phase shift calculater
    other_params = (method, tol_convergence, max_itr, xtol_root, rtol_root, atol_ode, rtol_ode)

    xsect_err = 1.0 # for error estimation

    # First, calculate L=0 and 1
    L = 0
    delta_arr = [phase_shift(a, b, L, sign, *other_params)]
    xsectk2_arr = [0.5 * np.sin(delta_arr[0])**2*(legendre_integ(0,0) - legendre_integ(1,0))]

    L = 1
    delta_arr.append(phase_shift(a, b, L, sign, *other_params))
    xsectk2_arr.append(1.5*inside_sum(1, 1, delta_arr[1], delta_arr[1]))

    # xsectk2 = np.sum(xsectk2_arr)
    #max_L = np.max([int(np.round(a*b)), 10])

    # Then iterate until convergence specified by tol
    for L in range(2, max_L+1, 1):
        
        #print "L=", L
        delta_arr.append(phase_shift(a, b, L, sign, *other_params))

        if L % 2 == 0:
            ls = np.arange(0, L ,2)
            part_wave = sum(list(map(lambda l: 0.5*inside_sum(L, l, delta_arr[L], delta_arr[l]), ls)))
            part_wave += sum(list(map(lambda l: 0.5*inside_sum(l, L, delta_arr[l], delta_arr[L]), ls)))
            part_wave += 0.5*inside_sum(L, L, delta_arr[L], delta_arr[L])
            xsectk2_arr.append(part_wave)

        else:
            ls = np.arange(1, L ,2)
            part_wave = sum(list(map(lambda l: 1.5*inside_sum(L, l, delta_arr[L], delta_arr[l]), ls)))
            part_wave += sum(list(map(lambda l: 1.5*inside_sum(l, L, delta_arr[l], delta_arr[L]), ls)))
            part_wave += 1.5*inside_sum(L, L, delta_arr[L], delta_arr[L])
            xsectk2_arr.append(part_wave)

        if L >= eval_L:
            xsectk2 = sum(xsectk2_arr[0:-eval_L])
            xsect_k2_del = sum(xsectk2_arr[-eval_L:])
            xsect_err = np.abs(xsect_k2_del/xsectk2)
            #print '     L=', L, 'delta_maxL=', delta_arr[-1], 'xsecterr=', xsect_err
            if xsect_err < tol and np.abs(delta_arr[-1]) < tol:
                break
    #print 'L=', L
    xsectk2 = sum(xsectk2_arr)
    
    return np.real(xsectk2), L, np.real(xsectk2_arr), delta_arr

def xsectk2_born(a, b, ab_th=1e-3):
    # Return sigma_Tk^2/4pi for Born approximation, valid for b << 1.
    # To avoid numerical issues, we separate computation by a*b value
    if a*b < ab_th:
        return a**2*b**4/4. - a**4*b**6 + 23*a**6*b**8/6.0
    else:
        return (6 * np.log(1+2*a**2*b**2) - (3. + 8*a**2*b**2)/(1. + 2*a**2*b**2) * np.log(1. + 4*a**2*b**2))/16.0/a**2


def xsect_clas(mX, mphi, alphaX, v, sign):
    """
    classical formula (1512.05344) valid for mX*v >> mphi.
    I have multiplied factor 0.5 (majorana_factor) to match the result of partial wave expansion.
    """
    majorana_factor = 0.5
    beta = 2*alphaX*mphi/mX/v**2
    if sign > 0:
        # attractive
        if beta < 1e-2:
            return 2*np.pi/mphi**2 * beta**2 * np.log(1 + 1.0/beta**2) * majorana_factor
        elif beta < 1e2:
            return 7*np.pi/mphi**2 * (beta**1.8 + 280*(beta/10)**10.3) / (1.0 + 1.4*beta + 0.006*beta**4 + 160.0*(beta/10.0)**10) * majorana_factor
        else: 
            return 0.81*np.pi/mphi**2 * (1.0 + np.log(beta) - 1.0/2.0/np.log(beta))**2 * majorana_factor

    # Tulin's parametrization
    # if sign > 0:
    #     if beta < 1e-1:
    #         return 4*np.pi/mphi**2 * beta**2 * np.log(1 + 1.0/beta) * majorana_factor
    #     elif 1e3:
    #         return 8*np.pi/mphi**2 * beta**2 / (1.0 + 1.5*beta**1.65) * majorana_factor
    #     else:
    #         return np.pi/mphi**2 * (np.log(beta) + 1 - 0.5/np.log(beta))**2 * majorana_factor
    else:
        print('Classical approx. for repulsice potential is not yet implemented...')
        return 0

def xsect(mX, mphi, alphaX, v, sign, minb=1e-2, classical_th = 10,
          tol=1e-2, eval_L=2, max_L=260,
          method='RK45', tol_convergence=0.01, max_itr=100,
          xtol_root=1e-4, rtol_root=1e-4,
          atol_ode=1e-4, rtol_ode=1e-5):

    """
    Return tranfer cross section in unit of mass.
    Only t+u channel
    (e.g., if mX and mphi are given by [GeV], xsect is [GeV^-2])

    mX: DM mass
    mphi: mediator mass
    alpha_X: yukawa coupling^2/4pi
    v: relative velocity
    sign: +1 for attractive, -1 for repulsive force
    minb: we use Born approximation for b < minb
    classical_th: for mX*v/mphi > classical_th, we use fitting formula (1512.05344)
                  Set 'classical_th=np.inf' to turn off the classical approximation.

    Other parameters are related to xsectk2_part_wave.
    """
    
    a = v/(2*alphaX)
    b = alphaX*mX/mphi

    other_params = (tol, eval_L,  max_L,
                    method, tol_convergence, max_itr,
                    xtol_root, rtol_root,
                    atol_ode, rtol_ode)

    if b < minb:
        return xsectk2_born(a, b)*16*np.pi/v**2/mX**2
    else:
        if mX*v/mphi < classical_th:
            return xsectk2_part_wave(a, b, sign, *other_params)[0]*16*np.pi/v**2/mX**2
        else:
            return xsect_clas(mX, mphi, alphaX, v, sign)


if __name__ == "__main__":
    # execute only if run as a script
    mX = 10.
    mphi = 1e-2
    alphaX = 1e-3
    sign = 1
    v = 1e-3

    res = xsect(mX, mphi, alphaX, v, sign)

    print(res)
