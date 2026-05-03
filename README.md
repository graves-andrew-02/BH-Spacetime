Here is a set of codes that use the ADM (3+1) Formalism to simulate a non-spinning black hole.


"Why do we need to do that?" - you, probably.


There is a massive amount of freedom in general relativity to choose your coordinates, in this type of code this presents as a freedom to choose how time and space progress over a coordinate grid. However, not many choices work for binary black hole simulations (all the stuff needed to get gravitational wave-forms), so this is a testbed that can be used to try out different coordinate conditions (generally called gauge conditions) in a non-trivial setting. The codes work for the case of spherical symmetry and so cannot produce a quadrapole moment and GW's. But it is cheap to run and relatively simple to code new conditions in comparison to BSSN codes, as this uses way fewer evolved variables.


As for the work of the codes, the "system" is about how I have decomposed the metric variables $\gamma_{rr}$ and $\gamma_{\theta \theta}$ into ones which are regular at the origin. Start with isotropic Schwarschild coordinates 

$ds^2 = -\frac{1-\frac{M}{2r}}{1+\frac{M}{2r}}dt^2 + (1+\frac{M}{2r})^4 (dr^2 + r^2 (d\theta^2 + sin^2 \theta d\phi^2))$

and chuck away the time time component as we work with spatial slices. System 1 takes out a factor of $\chi_0 = (1+\frac{M}{2r})^4$ the variables of the spatial metric $\tilde{a} = a/\chi_0$, $\tilde{b} = b/r^2\chi_0$ and for the extrinsic curvature $\tilde{c} = c/\chi_0$, $\tilde{d} = d/r^2\chi_0$. In normal notation $a = \gamma_{rr}$, $b = \gamma_{\theta \theta} = \gamma_{\phi \phi}/sin^2\theta$ and $c = K_{rr}$, $d = K_{\theta \theta} = K_{\phi \phi}/sin^2\theta$.


System 2 is the same as 1, accept evolves a variable $\Theta_b = \frac{\tilde{b}}{\tilde{a}}$ instead of $\tilde{b}$. This can lead to more stable simulaations when trying to form trumpet geometries via 1+log slicing and $\Gamma$-driver. 

System 3 is halfway to being a conformal traceless decomposition. Here I take out a factor of $\chi = (ab^2)^{1/3}$ from $a$ and $b$ to help form trumpet geometries. the extrinsic curvature is treated the same way as in system 1.


The analysis file is to take save files and produce plots of the constraints, relationship between the swartzchild radial coordinate and ours, and the proper distance from the outer event horizon on a fixed time slice. These give quite nice depictions on whats going on in the slices.
