# Convex Piecewise-Linear Fused Lasso
We solve the solution to a convex piecewise linear fused lasso problem, for all tradeoff parameter $\lambda > 0$, simultaneously!
This is a joint work with Cheng Lu (chenglu@berkeley.edu) and Dorit S. Hochbaum (hochbaum@ieor.berkeley.edu).

### Introduction

Piecewise-Linear Fused Lasso : we want to find $x$ that minimizes the following objective function: $\sum \limits_{i=1}^n f_i(x) + \lambda \sum \limits_{i=1}^{n-1} |x_i-x_{i+1}|$ [eq 1] where $f_i(x_i)$ is a convex piesewise linear function with respect to $x_i$.

Usually, a problem in this form needs parameter tuning for the ''best'' value of $\lambda$. However, a given list of values on which the parameter is tuned on might not contain (or that it might miss) some important values that lead to interesting results. Here, in this work, we overcome this issue by solving the solution for ALL nonnegative values of $\lambda$.

An example of a convex piecewise linear fused lasso is the following fused lasso problem with $\ell_1$ fidelity function. $\text{minimize} \sum \limits_{i=1}^n w_i |x_i-a_i| + \lambda \sum \limits_{i=1}^{n-1} |x_i-x_{i+1}|$. [eq 2]

### Solving $\ell_1$-fidelity fused lasso

We call the parameters $w$ and $a$ in [eq 2]  ''weights'' and ''breakpoints''. We use them as inputs to our solver.

```{python}
from tree_l1 import *
import numpy as np

## generate an example of weights array and breakpoints array
N = 10
np.random.seed(37)
weights = np.random.rand(N)
breakpoints = np.random.rand(N)

tree = Tree()
tree.fit(weights, breakpoints)
tree.full_split()

solutions = tree.all_solutions
lambda_ranges = tree.all_lambda_ranges
lambda_ranges_type = tree.all_lambda_ranges_type

# Explanation:
A lambda value in the interval (or range) of lambda_ranges[i] has a solution of solutions[i].
lambda_ranges_type[i] provides the type of the border of the interval, whether each end of the interval is included in or excluded from the interval.

# solutions look like this:
array([[0.37293743, 0.37293743, 0.37293743, 0.37293743, 0.37293743,
        0.37293743, 0.37293743, 0.37293743, 0.37293743, 0.37293743],
       [0.74926221, 0.74926221, 0.74926221, 0.74926221, 0.32556672,
        0.32556672, 0.32556672, 0.32556672, 0.1699427 , 0.1699427 ],
       [0.74926221, 0.74926221, 0.74926221, 0.74926221, 0.32556672,
        0.32556672, 0.32556672, 0.32556672, 0.32556672, 0.32556672],
       [0.74926221, 0.74926221, 0.74926221, 0.74926221, 0.37293743,
        0.37293743, 0.37293743, 0.37293743, 0.37293743, 0.37293743],
       [0.85801271, 0.85801271, 0.85801271, 0.85801271, 0.32556672,
        0.32556672, 0.37293743, 0.37293743, 0.1699427 , 0.1699427 ],
       [0.85801271, 0.85801271, 0.85801271, 0.85801271, 0.32556672,
        0.32556672, 0.32556672, 0.32556672, 0.1699427 , 0.1699427 ],
       [0.88680146, 0.85801271, 0.74926221, 0.87014472, 0.18675584,
        0.32556672, 0.37293743, 0.79371303, 0.15106027, 0.1699427 ],
       [0.88680146, 0.85801271, 0.74926221, 0.87014472, 0.18675584,
        0.32556672, 0.37293743, 0.37293743, 0.15106027, 0.1699427 ],
       [0.88680146, 0.85801271, 0.74926221, 0.87014472, 0.32556672,
        0.32556672, 0.37293743, 0.37293743, 0.15106027, 0.1699427 ],
       [0.88680146, 0.85801271, 0.85801271, 0.87014472, 0.32556672,
        0.32556672, 0.37293743, 0.37293743, 0.15106027, 0.1699427 ],
       [0.88680146, 0.85801271, 0.85801271, 0.87014472, 0.32556672,
        0.32556672, 0.37293743, 0.37293743, 0.1699427 , 0.1699427 ],
       [0.85801271, 0.85801271, 0.85801271, 0.87014472, 0.32556672,
        0.32556672, 0.37293743, 0.37293743, 0.1699427 , 0.1699427 ]])

# lambda_ranges look like this

array([[2.84370564,        inf],
       [1.37291106, 1.40497097],
       [1.40497097, 1.66489886],
       [1.66489886, 2.84370564],
       [0.4340016 , 0.55145476],
       [0.55145476, 1.37291106],
       [0.        , 0.10485758],
       [0.10485758, 0.19169039],
       [0.19169039, 0.36769865],
       [0.36769865, 0.37091382],
       [0.37091382, 0.42572141],
       [0.42572141, 0.4340016 ]])

# lambda_ranges_type look like this
array([['ex', 'ex'],
       ['ex', 'ex'],
       ['in', 'ex'],
       ['in', 'in'],
       ['ex', 'in'],
       ['ex', 'in'],
       ['in', 'in'],
       ['ex', 'ex'],
       ['in', 'ex'],
       ['in', 'ex'],
       ['in', 'in'],
       ['ex', 'in']], dtype='<U2')

```

### Solving general convex piecewise linear fused lasso


