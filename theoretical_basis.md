---
layout: default
title: Theoretical Foundations
---

# Theoretical Foundations

## Newtonian Gravity and Escape Velocity
The concept of an object with gravity so strong that light cannot escape was first considered in the 18th century, well before Einstein's theory. Using Newtonian mechanics:

$$
v_{\text{escape}} = \sqrt{\frac{2GM}{R}}
$$

If $$ v_{\text{escape}} \geq c $$ (the speed of light), even light cannot escape. For the sake of perspective, if we consider the Sun, this occurs when $$ R $$ is approximately 3 km.

## Einstein’s General Relativity (1915)
Albert Einstein’s field equations describe gravity as the curvature of spacetime caused by the distribution of mass-energy within it:

$$
R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}
$$

where:
- $$ R_{\mu\nu} $$ : Ricci curvature tensor
- $$ g_{\mu\nu} $$ : Metric tensor
- $$ R $$ : Ricci scalar
- $$ \Lambda $$ : Cosmological constant
- $$ T_{\mu\nu} $$ : Energy-momentum tensor

These equations predict phenomena like time dilation, the warping of spacetime, and most important for us, the formation of black holes.

## Schwarzschild Solution (1916)
Karl Schwarzschild provided the first exact solution to Einstein’s equations for a non-rotating, spherically symmetric massive body [1]. The same solution was again independently found by Johannes Droste, also in 1916. This solution, called the Schwarzschild metric, has the form:

$$
ds^2 = c^2d\tau^2 = \left( 1 - \frac{r_s}{r} \right)c^2dt^2 - \left( 1 - \frac{r_s}{r} \right)^{-1}dr^2 - r^2d\theta^2 - r^2\sin^2\theta d\phi^2
$$

The Schwarzschild radius $$ r_s $$ defines the event horizon, the boundary of a black hole:

$$
r_s = \frac{2GM}{c^2}
$$

For an object smaller than $$ r_s $$, its escape velocity exceeds $$ c $$, forming a black hole. The Schwarzschild Metric has 2 singularities: one at $$ r=0 $$ and one at the event horizon $$ r = r_s $$. Although physicists agreed that the singularity at $$ r = 0 $$ was a 'genuine' singularity which was supposed to be there, there was much debate over the meaning of the singularity at $$ r = r_s $$. It took until the 1960s to come up with a rigorous mathematical proof that the $$ r = r_s $$ singularity is the event horizon, which is a boundary in spacetime that can only be crossed in one direction.

<br>

<div style="display: flex; justify-content: space-between; margin-top: 20px;">
  <a href="/AST207project/index.html" style="text-decoration: none; font-weight: bold;">Back to Homepage</a>
  <a href="/AST207project/accretion_disk.html" style="text-decoration: none; font-weight: bold;">Next Page &#8594;</a>
</div>

<br>

