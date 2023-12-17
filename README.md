# Convex Piecewise-Linear Fused Lasso
We solve the solution to a convex piecewise linear fused lasso problem, for all tradeoff parameter $\lambda > 0$, simultaneously!

Piecewise-Linear Fused Lasso : we want to find $x$ that minimizes the following objective function: $\sum \limits_{i=1}^n f_i(x) + \lambda \sum \limits_{i=1}^{n-1} |x_i-x_{i+1}|$ where $f_i(x_i)$ is a convex piesewise linear function with respect to $x_i$.

Usually, a problem in this form needs parameter tuning for the ``best'' value of $\lambda$. However, a given list of values on which the parameter is tuned on might not contain (or that it might miss) some important values that lead to interesting results. Here, in this work, we overcome this issue by solving the solution for ALL nonnegative values of $\lambda$.

An example of a convex piecewise linear fused lasso is the following fused lasso problem with $\ell_1$ fidelity function. $\text{minimize} \sum \limits_{i=1}^n w_i |x_i-a_i| + \lambda \sum \limits_{i=1}^{n-1} |x_i-x_{i+1}|$.

This is a joint work with Cheng Lu (chenglu@berkeley.edu) and Dorit S. Hochbaum (hochbaum@ieor.berkeley.edu).
