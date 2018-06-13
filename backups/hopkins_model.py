# Hopkins model generation of light wavelengths
# Nouralishahi, Wu, Vandenberghe (2008). "Model Calibration for Optical
# Lithography via Semidefinite Programming."
def hopkins(m, sigma = 1):
	xi = np.linspace(0, m-1, num = m)
	
	def K(x):
		if x == 0:   # TODO: What is \lim_{x -> 0} J_1(x)/x?
			return 0
		return np.i0(np.abs(x))/np.abs(x)
	
	def J(x):
		return 2*K(sigma*x)
	
	W0 = np.zeros((m,m))
	for i in range(m):
		for j in range(m):
			W0[i,j] = K(xi[i])*np.conj(K(xi[j]))*J(xi[j] - xi[i])
	return W0
