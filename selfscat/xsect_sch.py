import numpy as np

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
