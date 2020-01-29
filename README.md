# self_scat_xsect
Self-scattering cross section of DM in the non-relativistic limit.
This module solves Schrodinger equation by the partial wave expansion.

This code calculates the t+u-channel cross section by partial wave expansion, and take velocity average of the resultant transfer cross section.

It also calculates the contribution from the s-channel exchange of a scalar particle, and the interference between s- and t+u-channel.

# Test code
In ```test``` directory, ```test.py``` computes the cross section for some model values, and calculate likelihood for the preferred cross section presented in [[Kaplinghat et al., 2016]](https://link.aps.org/doi/10.1103/PhysRevLett.116.041302).
The data is placed in ```data``` directory.

Just type ```python test.py``` in your command line.

For the minimal usage you can pass the following model parameters:

- ```mX```: DM mass [GeV]

- ```mphi```: mediator mass [GeV]

- ```alphaX```: Yukawa coupling

- ```vmean```: mean velocity of DM in natural unit

- ```sign```: +1 for attractive, -1 for repulsive potential

- ```width_sm```: decay width of the mediator [GeV], necessary to evaluate the s-channel contribution.

Inside the codes, you find many more parameters which already have default values (e.g., ```minn=10```).
They are relevant for precison of solving ODE, summing partial waves and averaging over the velocity.
I recommend using the default values.

# Inside ```xsect.py```
This file implements the cross section calculation.
You will mainly use the following velocity averaged cross section times velocity to calculate the likelihood:

- ```mean_xsectv(mX, mphi, alphaX, vmean, sign)```:
t+u-channel contribution. 

- ```mean_xsectv_sch(mX, mphi, alphaX, vmean, width_sm)```:
s-channel contribution.

- ```mean_xsectv_intf(mX, mphi, alphaX, vmean, sign, width_sm)```
interference between s- and t+u-channel.

Instead of using ```mean_xsectv``` and ```mean_xsectv_intf``` individually, you should use the following function for the efficient calculation:

- ```mean_xsectv_tu_intf(mX, mphi, alphaX, vmean, sign, width_sm)```: returns the sum of t+u-channel and interference contributions.

This function shares the phase shift among t+u-channel and interference term.

# Comments
Solving ODE works not so well for vely low ```b=mX*alphaX/mphi``` and for large velocity region.
In the formar case, we can just use the analytic expression of the Born approximation.
In the latter case, we may use fitting formula although I do not implement it for now.
