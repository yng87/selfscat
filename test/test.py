import os 
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../selfscat/'))
from likelihood import *

def test1():
    """
    t-channel resonance region
    """
    
    filepath = '../sample_data/fitted_xsect_log.dat'

    mX = 10
    mphi = 0.01
    alphaX = 0.00099
    sign = 1.0
    width_sm = 0.0

    b = alphaX*mX/mphi
    print("t-channel resonance region, b=", b)

    params = (mX, mphi, alphaX, sign, width_sm)

    likelihood_HB_data(filepath, *params)

def test2():
    """
    s-channel resonance region.
    P-wave benchmark of Murayama-san's paper
    """
    filepath = '../sample_data/fitted_xsect_log.dat'

    c = 3e5
    mtilde = 0.4 #GeV
    vR = 108 / c 
    gamma = 1e-3

    mX = mtilde * (0.25)**(1.0/3)
    mphi = np.sqrt(vR**2 + 4) * mX
    alphaX = 8*mphi**2*gamma/mX**2 
    print(mX, mphi, alphaX, mphi - mX*2)
    b = alphaX*mX/mphi
    print("s-channel resonance region, b=", b)

    sign = 1.0
    width_sm = 0.0

    params = (mX, mphi, alphaX, sign, width_sm)

    likelihood_HB_data(filepath, *params)


def test3():
    """ 
    Born region
    """
    filepath = '../sample_data/fitted_xsect_log.dat'

    mX = 0.1
    mphi = 1e-4
    alphaX = 7e-6
    sign = 1.0
    width_sm = 0.0

    b = alphaX*mX/mphi
    print("Born region, b=", b)
    params = (mX, mphi, alphaX, sign, width_sm)
    likelihood_HB_data(filepath, *params)


def main():
    test1()
    test2()
    test3()
    
if __name__ == '__main__':
    main()
