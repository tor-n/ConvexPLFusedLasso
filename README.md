# Convex Piecewise-Linear Fused Lasso
We solve the solution to a convex piecewise linear fused lasso problem, for all tradeoff parameter $\lambda > 0$, simultaneously!

Piecewise-Linear Fused Lasso : we want to find $x$ that minimizes the following objective function: $\sum \limits_{i=1}^n f_i(x) + \lambda \sum \limits_{i=1}^{n-1} |x_i-x_{i+1}|$ where $f_i(x_i)$ is a convex piesewise linear function with respect to $x_i$.

In this work, we generate all solutions for all nonnegative values of $\lambda$.

This is a joint work with Cheng Lu (chenglu@berkeley.edu) and Dorit S. Hochbaum (hochbaum@ieor.berkeley.edu).
