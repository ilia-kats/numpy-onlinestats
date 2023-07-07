# numpy-onlinestats

This is a Python package for element-wise streaming statistics of Numpy arrays, meaning that arrays can be added one-by-one.
This is much more memory-efficient than first collecting all arrays before calculating statstics.
One major usecase is Bayesian modeling, where the posterior distribution is often intractable and can only be approximated via sampling.
This concerns both MCMC and variational inference meethods. MCMC is inherently sampling-based, while variational inference methods can have derived quantities or structured posteriors that do not admit closed-form expressions for properties of their distribution.

numpy-onlinestats approximates quantiles and cumulative distribution functions using the [t-digest algorithm](http://arxiv.org/abs/1902.04023) (in particular, it uses [this implementation](https://github.com/SpirentOrion/digestible)) and calculates exact moments using a [numerically stable algorithm](https://www.johndcook.com/blog/skewness_kurtosis/).

## Requirements

- Python 3.10 or newer
- A C++20 compatible compiler (developed with GCC 13 using `-std=c++20`)

## Sample code

```python
import numpy as np
import numpy_onlinestats as npo

stats = npo.NpOnlineStats(np.random.uniform((5, 3, 7)))
for i in range(100):
    stats.add(np.random.uniform((5, 3, 7)))

stats.quantile(0.25)
stats.mean()
stats.var()
```
