#TRS

The Solver for Trust Region Subproblem

## Description
It is the solver that handles with the (generalized) trust region subproblem.  
It transpalents MATLAB `trust()` method by Python 2.x.
Please see ([File Exchange - MATLAB Central](http://www.mathworks.com/matlabcentral/fileexchange/28013-rosin-rammler-diagram-plotting-tool/content/RRD%20-%20ln/funct/trust.m))

### (Generalized) Trust Region Subproblem
(Generalized) trust region subproblem is the following mathematical optimization problem:  
  
![Not Exists!](http://i.imgur.com/hr7gYq4.png "(Generalized) Trust Region Subproblem")  
  
where ![Not Exists!](http://i.imgur.com/0CurIWu.png?1 "Given Data") are the given data.If ![Not Exists!](http://i.imgur.com/yBsDWAT.png?1) is a positive semidefinite matrix, then the above problem is nonconvex.But We suppose that ![Not Exists!](http://i.imgur.com/yBsDWAT.png?1) consider a generalized matrix.

## Requirement
[NumPy](http://www.numpy.org/) only!

## Install
You can download the `TRS.py`, then import it as

```python
>>> from TRS import trust
```

in the Python source code that places the same directory of `TRS.py`.

## Usage
You may use `trust(g, H, Delta)` as the solver.

+ Arguments as `g, H, Delta` correspond with the above given data
+ Returns are `s, val, time, count, lmbd` where
	- `s`		: The optimal solution
	- `val`		: The optimal value
	- `time`	: The calculation time(ms)
	- `count`	: The number of iterations
	- `lmbd`	: The corresponding Lagrange multiplier

### Example
![Not Exists!](http://i.imgur.com/OcvDNjQ.png "Example").  

```python
>>> import numpy as np
>>> from math import sqrt
>>> from TRS import trust
>>> g = np.array([-5, 1])
>>> H = np.array([[8, -3], [-3, -2]])
>>> Delta = sqrt(2.0)
>>> s, val, time, count, lmbd = trust(g, H, Delta)
------Trust Region Subproblem------
Solution            : [ 0.77053554  1.18586465]
Value               : -4.4394405408
Time(ms)            : 4.591
Iteration Count     : 9
Lagrange multiplier : 3.10603400638
-> Finished!!
>>> s
array([ 0.77053554,  1.18586465])
>>> count
9
```

## Licence
Kido Takeru([@infhyroyage](https://twitter.com/infhyroyage))
