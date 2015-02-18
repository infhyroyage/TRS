'''
Created on 2014/06/21

@author: Kido Takeru
'''

# import
import numpy as np
from math import sqrt, isnan

'''
[seceqn]
Return the scaler of the secular equation at the scaler lmbd
'''
def seceqn(lmbd, eigval, alpha, delta):
	M = eigval + lmbd
	zeroidx = np.where(M == 0.0)
	MM = alpha
	M[M != 0.0] = MM[M != 0.0] / M[M != 0.0]
	M[zeroidx] = float("Inf")
	M = M ** 2
	value = sqrt(1.0 / np.sum(M))
	if (isnan(value)):
		value = 0.0
	
	return 1.0 / delta - value

'''
[rfzero]
Find zero of the finction seceqn(lmbd, eigval, alpha, delta) to the RIGHT of the starting point of the lmbd as x.
A small modification of the M-file fzero to ensure a zero to the Right of x is searched for.
[rfzero] is a slightly modified version of function FZERO.
FZERO(F,X) finds a zero of F(X).
F is a string containing the name of a real-valued function of a single real variable.
X is a starting guess.
The value returned is near a point where F changes sign.
For example, FZERO('sin',3) is pi.
Note the quotes around sin.
Ordinarily, functions are defined in M-files.

An optional third argument sets the relative tolerance for the convergence test.
'''
def rfzero(laminit, eigval, alpha, delta, itbnd = 50, tol = 1e-12):
	# Initialization
	itfun = 0
	
	if (laminit != 0.0):
		dx = abs(laminit) / 2.0
	else:
		dx = 0.5
	
	a = laminit
	c = a
	fa = seceqn(a, eigval, alpha, delta)
	itfun += 1
	
	b = laminit + dx
	fb = seceqn(b, eigval, alpha, delta)
	itfun += 1
	
	# Find change of sign
	while ((fa > 0.0) == (fb > 0.0)):
		dx *= 2.0
		'''
		if ((fa > 0.0) != (fb > 0.0)):
			break
		'''
		b = laminit + dx
		fb = seceqn(b, eigval, alpha, delta)
		itfun += 1
		if (itfun > itbnd):
			break
	
	fc = fb
	d = b - a
	e = d	
	
	# Main loop, exit from middle of the loop
	while (fb != 0.0):
		# Insure that b is the best result so far, a is the previous value of b, and c is on the opposite of the zero from b.
		if ((fb > 0.0) == (fc > 0.0)):
			c = a
			fc = fa
			d = b - a
			e = d
		
		if (abs(fc) < abs(fb)):
			a = b
			b = c
			c = a
			fa = fb
			fb = fc
			fc = fa
		
		# Convergence test and possible exit
		if (itfun > itbnd):
			break
		
		m = 0.5 * (c - b)
		toler = 2.0 * tol * max(abs(b), 1.0)
		if ((abs(m) <= toler) or (fb == 0.0)):
			break
		
		# Choose bisection or interpolation
		if ((abs(e) < toler) or (abs(fa) <= abs(fb))):
			# Bisection
			d = m
			e = m
		else:
			# Interpolation
			s = fb / fa
			
			if (a == c):
				# Linear interpolation
				p = 2.0 * m * s
				q = 1.0 - s
			else:
				# Inverse quadratic interpolation
				q = fa / fc
				r = fb / fc
				p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
				q = (q - 1.0) * (r - 1.0) * (s - 1.0)
			
			if (p > 0.0):
				q = -q
			else:
				p = -p
			
			# Is interpolated point acceptable
			if ((2.0 * p < 3.0 * m * q - abs(toler * q)) and (p < abs(0.5 * e * q))):
				e = d
				d = p / q
			else:
				d = m
				e = m
		
		# Next point
		a = b
		fa = fb
		
		if (abs(d) > toler):
			b += d
		else:
			if (b > c):
				b = b - toler
			else:
				b += toler
		
		fb = seceqn(b, eigval, alpha, delta)
		itfun += 1
	
	return b, itfun

'''
[trust]
Solves the trust region problem:
MINIMIZE	: g^T s + 1/2 s^T H s
SUBJECT TO	: ||s|| <= delta.

<input>
g			: The above n dimension vertor
H			: The above n * n square symmetric matrix
delta		: The above scaler

<return>
s, val
where
s			: The solution of the trust region problem given by the secular equation: 1/delta - 1/(||s||) = 0.
val			: The value of the quadratic at the solution
'''

def trust(g, H, delta, tolvval = 1e-8, eps = 2 ** -52):
	print "------Trust Region Subproblem------"
	
	# Set initial constant parameters
	key = 0
	lmbd = 0.0
	laminit = 0.0
	
	'''
	Determine the variables
	where
	coeff, alpha	: The n dimension vertor
	eigval			: The n dimension vector whose elements are the eigenvalues of the input matrix H
	eigvec			: The n * n square matrix whose whose column elements are the eigenvalues of the input matrix H related with eigval index
	mineigval		: The minimum value of eigval
	argmineigval	: The index of mineigval
	sig				: The scaler
	''' 
	coeff = np.zeros(g.size)
	eigval, eigvec = np.linalg.eig(H)
	mineigval= np.min(eigval)
	argmineigval = np.argmin(eigval)
	alpha = np.dot(eigvec.T, -g)
	sig = np.sign(alpha[argmineigval]) + (alpha[argmineigval] == 0.0)
	
	# Generate n dimension vector as initial solution
	s = np.zeros(g.size)
	
	# POSITIVE DEFINITE CASE
	if (mineigval > 0.0):
		coeff = alpha / eigval
		s = np.dot(eigvec, coeff)
		# Next 1.0 is mysterious...
		if (np.linalg.norm(s) <= 1.0 * delta):
			key = 1
	else:
		laminit = -mineigval
	
	# INDEFINITE CASE
	if (key == 0):
		if (seceqn(laminit, eigval, alpha, delta) > 0.0):
			b, _ = rfzero(laminit, eigval, alpha, delta)
			vval = abs(seceqn(b, eigval, alpha, delta))
			
			if (vval <= tolvval):
				lmbd = b
				key = 2
				lam = lmbd * np.ones(g.size)
				w = eigval + lam
				coeff[w != 0.0] = alpha[w != 0.0] / w[w != 0.0]
				coeff[np.all([w == 0.0, alpha == 0.0], axis = 0)] = 0.0
				coeff[np.all([w == 0.0, alpha != 0.0], axis = 0)] = float("Inf")
				coeff[np.isnan(coeff)] = 0.0
				s = np.dot(eigvec, coeff)
				if ((np.linalg.norm(s) > 1.2 * delta) or (np.linalg.norm(s) < 0.8 * delta)):
					key = 5
					lmbd = -mineigval
			else:
				lmbd = -mineigval
				key = 3
		else:
			lmbd = -mineigval
			key = 4
		
		lam = lmbd * np.ones(g.size)
		
		if (key > 2):
			# KIMOI code
			#arg = abs(eigval + lam) < 10.0 * eps * max(abs(eigval), 1.0)
			#alpha[arg] = 0.0
			alpha[abs(eigval + lam) < 10.0 * eps * max(abs(eigval), 1.0)] = 0.0
		
		w = eigval + lam
		coeff[w != 0.0] = alpha[w != 0.0] / w[w != 0.0]
		coeff[np.all([w == 0.0, alpha == 0.0], axis = 0)] = 0.0
		coeff[np.all([w == 0.0, alpha != 0.0], axis = 0)] = float("Inf")
		coeff[np.isnan(coeff)] = 0.0
		s = np.dot(eigvec, coeff)
		nrms = np.linalg.norm(s)
		
		if ((key > 2) and (nrms < 0.8 * delta)):
			s += sqrt(delta ** 2 - nrms ** 2) * sig * eigvec[:, argmineigval]
		
		if ((key > 2) and (nrms > 1.2 * delta)):
			b, _ = rfzero(laminit, eigval, alpha, delta)
			lmbd = b
			lam = lmbd * np.ones(g.size)
			w = eigval + lam
			coeff[w != 0.0] = alpha[w != 0.0] / w[w != 0.0]
			coeff[np.all([w == 0.0, alpha == 0.0], axis = 0)] = 0.0
			coeff[np.all([w == 0.0, alpha != 0.0], axis = 0)] = float("Inf")
			coeff[np.isnan(coeff)] = 0.0
			s = np.dot(eigvec, coeff)
	
	print "Solution :", s
	print "Radius   :", np.sqrt(np.linalg.norm(s) ** 2)
	print "Value    :", np.dot(g, s) + 0.5 * np.dot(s, np.dot(H, s))
	print "-> Finished!!\n"
	
	return s, np.dot(g, s) + 0.5 * np.dot(s, np.dot(H, s))

'''
[trustKai]
Solves the trust region problem:
MINIMIZE	: g^T s + 1/2 s^T H s
SUBJECT TO	: ||s|| <= delta.

<input>
g			: The above n dimension vertor
H			: The above n * n square symmetric matrix
delta		: The above scaler

<return>
s, val, count, lmbd
where
s			: The solution of the trust region problem given by the secular equation: 1/delta - 1/(||s||) = 0.
val			: The value of the quadratic at the solution
count		: The number of iterations of the above secular equation
lmbd		: The value of the corresponding Lagrange multiplier
'''

def trustKai(g, H, delta, tolvval = 1e-8, eps = 2 ** -52):
	print "------Trust Region Subproblem------"
	
	# Set initial constant parameters
	key = 0
	count = 0
	lmbd = 0.0
	laminit = 0.0
	
	'''
	Determine the variables
	where
	coeff, alpha	: The n dimension vertor
	eigval			: The n dimension vector whose elements are the eigenvalues of the input matrix H
	eigvec			: The n * n square matrix whose whose column elements are the eigenvalues of the input matrix H related with eigval index
	mineigval		: The minimum value of eigval
	argmineigval	: The index of mineigval
	sig				: The scaler
	''' 
	coeff = np.zeros(g.size)
	eigval, eigvec = np.linalg.eig(H)
	mineigval= np.min(eigval)
	argmineigval = np.argmin(eigval)
	alpha = np.dot(eigvec.T, -g)
	sig = np.sign(alpha[argmineigval]) + (alpha[argmineigval] == 0.0)
	
	# Generate n dimension vector as initial solution
	s = np.zeros(g.size)
	
	# POSITIVE DEFINITE CASE
	if (mineigval > 0.0):
		coeff = alpha / eigval
		s = np.dot(eigvec, coeff)
		# Next 1.0 is mysterious...
		if (np.linalg.norm(s) <= 1.0 * delta):
			key = 1
	else:
		laminit = -mineigval
	
	# INDEFINITE CASE
	if (key == 0):
		if (seceqn(laminit, eigval, alpha, delta) > 0.0):
			b, count = rfzero(laminit, eigval, alpha, delta)
			vval = abs(seceqn(b, eigval, alpha, delta))
			
			if (vval <= tolvval):
				lmbd = b
				key = 2
				lam = lmbd * np.ones(g.size)
				w = eigval + lam
				coeff[w != 0.0] = alpha[w != 0.0] / w[w != 0.0]
				coeff[np.all([w == 0.0, alpha == 0.0], axis = 0)] = 0.0
				coeff[np.all([w == 0.0, alpha != 0.0], axis = 0)] = float("Inf")
				coeff[np.isnan(coeff)] = 0.0
				s = np.dot(eigvec, coeff)
				if ((np.linalg.norm(s) > 1.2 * delta) or (np.linalg.norm(s) < 0.8 * delta)):
					key = 5
					lmbd = -mineigval
			else:
				lmbd = -mineigval
				key = 3
		else:
			lmbd = -mineigval
			key = 4
		
		lam = lmbd * np.ones(g.size)
		
		if (key > 2):
			# KIMOI code
			#arg = abs(eigval + lam) < 10.0 * eps * max(abs(eigval), 1.0)
			#alpha[arg] = 0.0
			alpha[abs(eigval + lam) < 10.0 * eps * max(abs(eigval), 1.0)] = 0.0
		
		w = eigval + lam
		coeff[w != 0.0] = alpha[w != 0.0] / w[w != 0.0]
		coeff[np.all([w == 0.0, alpha == 0.0], axis = 0)] = 0.0
		coeff[np.all([w == 0.0, alpha != 0.0], axis = 0)] = float("Inf")
		coeff[np.isnan(coeff)] = 0.0
		s = np.dot(eigvec, coeff)
		nrms = np.linalg.norm(s)
		
		if ((key > 2) and (nrms < 0.8 * delta)):
			s += sqrt(delta ** 2 - nrms ** 2) * sig * eigvec[:, argmineigval]
		
		if ((key > 2) and (nrms > 1.2 * delta)):
			b, count = rfzero(laminit, eigval, alpha, delta)
			lmbd = b
			lam = lmbd * np.ones(g.size)
			w = eigval + lam
			coeff[w != 0.0] = alpha[w != 0.0] / w[w != 0.0]
			coeff[np.all([w == 0.0, alpha == 0.0], axis = 0)] = 0.0
			coeff[np.all([w == 0.0, alpha != 0.0], axis = 0)] = float("Inf")
			coeff[np.isnan(coeff)] = 0.0
			s = np.dot(eigvec, coeff)
	
	print "Solution            :", s
	print "Radius              :", np.sqrt(np.linalg.norm(s) ** 2)
	print "Value               :", np.dot(g, s) + 0.5 * np.dot(s, np.dot(H, s))
	print "Iteration Count     :", count
	print "Lagrange multiplier :", lmbd
	print "-> Finished!!\n"
	
	return s, np.dot(g, s) + 0.5 * np.dot(s, np.dot(H, s)), count, lmbd
