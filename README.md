# Convex Piecewise-Linear FusedLasso
We solve the solution to a convex piecewise linear fused lasso problem, for all tradeoff parameter $\lambda > 0$, simultaneously!

Piecewise-Linear Fused Lasso : we want to find $x$ that minimizes the following objective function: $\sum_{i=1}^n f_i(x) + \lambda \sum_{i=1}^{n-1} |x_i-x_{i+1}|$.

In this work, we find generate all solutions for all nonnegative values of $\lambda$.
