#!/usr/bin/env python
import numpy

##########  Convenience functions: product, adjoint, commutator

def  product( A, B ):
# Convenience function: do matrix multiplication A*B
	return numpy.dot(A,B)

def adjoint( M ):
# Convenience function: find the adjoint of M
	return numpy.conjugate(numpy.transpose(M))

def commutator( A, B ):
# Convenience function : Find the commutator of two square NumPy arrays
	return numpy.dot(A,B) - numpy.dot(B,A)
	
	
##########	LINMAX utilities: makearray, maketimes, evolve

def makearray( list_or_array, user_type=None ):
# Convert list_or_array into NumPy array.
# Inputs: list or NumPy array, string name of datatype
# Outputs: NumPy array

	# Dictionary of NumPy datatypes
	datatype_dict = {
		'float32' : numpy.dtype(numpy.float32),
		'float64' : numpy.dtype(numpy.float64),
		'float128' : numpy.dtype(numpy.float128),
		'complex64' : numpy.dtype(numpy.complex64),
		'complex128' : numpy.dtype(numpy.complex128),
	 	}

	# Did user input a valid datatype?
	if user_type in datatype_dict:
		datatype = datatype_dict[user_type]
		new_array = numpy.array(list_or_array).astype(datatype)

	# If not, then let NumPy decide. (Multiply by 1.0 to cast int as float64.)
	else: 
		new_array = 1.0 * numpy.array(list_or_array)

	return new_array


def maketimes( t_start, t_stop, **kwargs):
# Make a 1D NumPy array of equally-spaced sample times.
# Options: numsteps, stepsize
# Note: J timesteps means (J+1) sample times. Datatype is always float64.
# Using t_start > t_stop is OK! (If so, stepsize should be negative.)
	
	# Assign optional variables as needed
	J = kwargs.get('numsteps',100)
	h = kwargs.get('stepsize')
	time_interval = 1.0*t_stop - 1.0*t_start

	# If user specified a stepsize, then use it.
	if h :
		J = numpy.floor( time_interval / h )
		times = t_start + h * numpy.arange(J+1)
	
	# Else use linspace to make sample times.
	else :
		times = numpy.linspace(t_start,t_stop,J+1)
	
	return times


def evolve( initial_state, S_matrices ):
# Evolve a state by acting step-forward matrices on it
# Inputs: initial state, list of step-foward matrices (as square NumPy arrays)
# Outputs: 2D NumPy array whose columns are new state vectors
# Note: initial_state can be list, 1D array, 2D row array, or 2D column array
	
	# Use same datatype as 0th S_matrix
	datatype = S_matrices[0].dtype
	
	# Get dimensions and initialize list of states
	N = len( initial_state )
	J = len( S_matrices )
	x_old = makearray(initial_state).astype(datatype)
	evolved_states = numpy.zeros([N,J+1]).astype(datatype)
	evolved_states[:,0] = x_old.ravel()		# Zeroth column is initial state
	
	# Act step-forward matrices on state vectors
	for j in range(0,J):
		x_new = numpy.dot(S_matrices[j],x_old)
		evolved_states[:,j+1] = x_new
		x_old = x_new
		
	return evolved_states
	

##########	Matrix exponentials: powerexp, selfadjointexp, antiadjointexp

def powerexp( M ):
# Exponentiate a square NumPy array by scaling, Taylor series, and squaring

	# Users may wish to adjust these numbers for speed vs. accuracy
	SMALL = 0.25;			# Rescale M if norm(M) > SMALL
	VERY_SMALL = 1e-12;		# Ignore terms smaller than norm(M)*VERY_SMALL

	# Find Frobenius norm of M and rescale it if needed
	q = 0;
	M_norm = numpy.linalg.norm(M,'fro');
	if ( M_norm > SMALL ):
		q = 2 + int( numpy.ceil( numpy.log2(M_norm) ) );
		M = M * numpy.exp2(-q);
	
	# Calculate Taylor terms through k=10 if needed
	new_term = M
	identity_matrix = numpy.identity(M.shape[0]).astype(M.dtype)
	exp_M = identity_matrix + new_term
	for k in range(2,11):
		new_term = numpy.dot( M / k , new_term)
		exp_M = exp_M + new_term
		
		# Check Frobenius norm of 4th, 6th, and 8th terms
		# If it is very small, stop calculating terms
		if (k==4) or (k==6) or (k==8) :
			if numpy.linalg.norm(new_term) < VERY_SMALL :
				break
				
	# Un-scale exp_M by squaring it q times (if q >0 )
	for j in range(q):
		exp_M = numpy.dot(exp_M,exp_M)
	
	return exp_M
	
		
def selfadjointexp( H ):
# Exponentiate a self-adjoint square NumPy array by diagonalization
# Note: uses numpy.linalg, which does not work with float128 datatype!
	
	# Find eigenvalues and eigenvectors
	eigvalues , U = numpy.linalg.eigh(H)
	
	# Exponentiate eigenvalues and construct a diagonal matrix
	eigvalues = numpy.exp(eigvalues)
	D = numpy.diag(eigvalues)
	
	# Calculate exponential of H
	Udag = adjoint(U)
	exp_H = numpy.dot( U , numpy.dot(D,Udag) ) 
	return exp_H
	
		
def antiadjointexp( A ):
# Exponentiate an anti-adjoint square NumPy array by diagonalization
# Note: uses numpy.linalg, which does not work with float128 datatype!

	# Define a self-adjoint matrix and diagonalize it
	H = 1.0j * A
	eigvalues , U = numpy.linalg.eigh(H)
	
	# Imaginary-exponentiate eigenvalues
	eigvalues = numpy.exp( -1.0j * eigvalues )
	D = numpy.diag(eigvalues)
	
	# Calculate exponential of A.
	Udag = adjoint(U)
	exp_A = numpy.dot( U, numpy.dot(D,Udag) )
	
	# If A is real, then discard erroneous imaginary parts of exp_A.
	if (A.dtype != numpy.complex64) and (A.dtype != numpy.complex128):
		exp_A = exp_A.real
		
	return exp_A


##########	Solvers: solveonce, generate, linmax4s, linearRK4

def solveonce( G, times, initial_state, order=6 ):
# Evolve a state without saving step-forward matrices
# G must be a function which inputs time and outputs a square NumPy array.
# Outputs: 2D NumPy array whose columns are state vectors

	# Sample G at 0th sample time to determine datatype
	datatype = G(times[0]).dtype
	
	# Constants used in LINMAX4
	sqrt3over6 = numpy.sqrt(3).astype(datatype) / 6.0
	c4L = 0.5 - sqrt3over6
	c4R = 0.5 + sqrt3over6
	
	# Constants used in LINMAX6
	sqrt15 = numpy.sqrt(15).astype(datatype)
	c6L = 0.5 - 0.1*sqrt15
	c6R = 0.5 + 0.1*sqrt15
	
	# Initialize x_old and new_states
	N = len(initial_state)
	J = len(times) - 1
	initial_state = numpy.array(initial_state).astype(datatype)
	new_states = numpy.zeros([N,J+1]).astype(datatype)
	x_old = initial_state.ravel()    # Reshape initial_state if needed
	new_states[:,0] = x_old
		
	# Generate a new state vector for each timestep
	for j in range(0,J):
		
		# Find stepsize (useful for custom timesteps)
		t_old = times[j]
		h = times[j+1] - t_old
		
		# Calculate a Magnus matrix with LINMAX2, LINMAX4, or LINMAX6		
		if order==2 :		# LINMAX2 algorithm
			Omega = h * G(t_old + 0.5*h)
			
		elif order==4 :		# LINMAX4 algorithm
			G1 = G( t_old + c4L*h )
			G2 = G( t_old + c4R*h )
			Omega = h/2.0*( G1 + G2 - h*sqrt3over6*commutator(G1,G2) )
			
		elif order==6 :		# LINMAX6 algorithm
			A_left = h * G(t_old + c6L*h)
			B1 = h * G(t_old + 0.5*h)
			A_right = h * G(t_old + c6R*h)
			B2 = sqrt15/3.0 * (A_right - A_left)
			B3 = 10.0/3.0 * (A_left - 2.0*B1 + A_right)
			C1 = commutator(B1,B2)
			D = 2.0*B3 + C1
			C2 = commutator(B1,D)
			E = -20.0*B1 - B3 + C1
			F = B2 - (1.0/60.0)*C2
			Omega = B1 + (1.0/12.0)*B3 + (1.0/240.0)*commutator(E,F)
			
		# Calculate new states and save as column of new_states
		x_new = numpy.dot( powerexp(Omega), x_old )
		new_states[:,j+1] = x_new
		
		# Update x_old for next timestep
		x_old = x_new

	return new_states


def generate( G, times, order=6 ):
# Generate step-forward matrices using LINMAX2, LINMAX4, or LINMAX6.
# G must be a function which inputs time and outputs a square NumPy array.
# Outputs: list of step-forward matrices (as square NumPy arrays)

	# Sample G at 0th sample time to determine datatype
	datatype = G(times[0]).dtype
	
	# Constants used in LINMAX4
	sqrt3over6 = numpy.sqrt(3).astype(datatype) / 6.0
	c4L = 0.5 - sqrt3over6
	c4R = 0.5 + sqrt3over6
	
	# Constants used in LINMAX6
	sqrt15 = numpy.sqrt(15).astype(datatype)
	c6L = 0.5 - 0.1*sqrt15
	c6R = 0.5 + 0.1*sqrt15
	
	# Generate a step-forward matrix for each timestep
	J = len(times) - 1
	S_matrices = [ ]
	for j in range(0,J):

		# Find stepsize (useful for custom timesteps)
		t_old = times[j]
		h = times[j+1] - t_old
		
		# Calculate a Magnus matrix with LINMAX2, LINMAX4, or LINMAX6		
		if order==2 :		# LINMAX2 algorithm
			Omega = h * G(t_old + 0.5*h)
			
		elif order==4 :		# LINMAX4 algorithm
			G1 = G( t_old + c4L*h )
			G2 = G( t_old + c4R*h )
			Omega = h/2.0*( G1+G2 - h*sqrt3over6*commutator(G1,G2) )
			
		elif order==6 :		# LINMAX6 algorithm
			A_left = h * G(t_old + c6L*h)
			B1 = h * G(t_old + 0.5*h)
			A_right = h * G(t_old + c6R*h)
			B2 = sqrt15/3.0 * (A_right - A_left)
			B3 = 10.0/3.0 * (A_left - 2.0*B1 + A_right)
			C1 = commutator(B1,B2)
			D = 2.0*B3 + C1
			C2 = commutator(B1,D)
			E = -20.0*B1 - B3 + C1
			F = B2 - (1.0/60.0)*C2
			Omega = B1 + (1.0/12.0)*B3 + (1.0/240.0)*commutator(E,F)
			
		# Exponentiate Magnus matrix
		S_matrices.append( powerexp(Omega) )

	return S_matrices


def linmax4s( G, times ):
# Generate step-forward matrices using LINMAX4S.
# G must be a function which inputs time and outputs a square NumPy array.
# Outputs: list of step-forward matrices (as square NumPy arrays)
	
	# Initialize S_matrices and G_old
	S_matrices = [ ]
	G_old = G(times[0])
	
	# Generate a step-forward matrix for each timestep
	J = len(times) - 1	
	for j in range(0,J):
		
		# Find stepsize (useful for custom timesteps)
		t_old = times[j]
		h = times[j+1] - t_old
		
		# Evaluate G at subsample times
		G_mid = G(t_old + 0.5*h)
		G_new = G(t_old + h)
		
		# Calculate a step-forward matrix
		Omega = h/6.0*( G_old + 4.0*G_mid + G_new - h/2.0*commutator(G_old,G_new) )
		S_matrices.append( powerexp(Omega) )
		
		# Recycle G_new for next timestep
		G_old = G_new
		
	return S_matrices
	
	
def linearRK4( G, times, **kwargs ):
# Generate step-forward matrices using linearized RK4 (Simpson) method.
# G must be a function which inputs time and outputs a square NumPy array.
# Outputs: list of step-forward matrices (as square NumPy arrays)

	# Initialize S_matrices and G_old
	S_matrices = []
	G_old = G(times[0])
	identity_matrix = numpy.identity(G_old.shape[0]).astype(G_old.dtype)

	# Generate a step-forward matrix for each timestep
	J = len(times) - 1
	for j in range(0,J):

		# Find stepsize (useful for custom timesteps)
		t_old = times[j]
		h = times[j+1] - t_old

		# Evaluate G at subsample times
		G_mid = G(t_old + 0.5*h)
		G_new = G(t_old + h)

		# Calculate these to avoid repeated matrix multiplication
		G_midold = numpy.dot(G_mid,G_old)
		G_newmid = numpy.dot(G_new,G_mid)

		# Calculate a step-forward matrix
		S_matrix = identity_matrix + h/6.0 *(
			G_old + 4*G_mid + G_new 
			+ h * (G_midold + numpy.dot(G_mid,G_mid) + G_newmid)
			+ h*h/2.0 * (numpy.dot(G_mid,G_midold) + numpy.dot(G_newmid,G_mid))
			+ h*h*h/4.0 * numpy.dot(G_newmid,G_midold)
			)
		S_matrices.append( S_matrix )

		# Recycle G_new for next timestep
		G_old = G_new

	return S_matrices