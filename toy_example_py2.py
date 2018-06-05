import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
p_num = 5
r_eps = 1e-4
# S = np.random.randint(0, 2, size = (5,5))
S = np.loadtxt(open("data/smiley_array.txt", "rb"), delimiter=",")

k,l = S.shape
n = np.prod(S.shape)
S_vec = S.flatten()   # Flatten into vector

# Sample points on a unit sphere
def samp_sphere(npoints, ndim):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis = 0)
    return vec

# Binary assignment with separating hyperplane drawn uniformly from unit sphere
# See Duchi's EE364B SDP relaxation slides, pg. 17
def rand_sign(Z):
	n = Z.shape[0]
	print(np.linalg.eigvalsh(Z))
	L = np.linalg.cholesky(Z)   # BUG: Z is not positive definite due to numerical error
	u = samp_sphere(1,n)
	return np.sign(L.T.dot(u))

# Normalized light intensity
def intensity(z, Z, Phi):
	ones = np.ones((n,1))
	term1 = diag(Phi.H*Z*Phi)
	term2 = np.diag(np.real(Phi.H.dot(ones).dot(ones.T).dot(Phi)))
	term3 = 2*diag(Phi.H*z*ones.T*Phi)
	return (term1 + term2 + term3)/4

# Generate light wavelengths
v = np.arange(n)
p_light = []
for i in range(p_num):
	data = np.random.randn(n,2)
	comp = np.apply_along_axis(lambda args: [complex(*args)], 1, data)
	bessel = scipy.special.jv(1,v).reshape((n,1))
	p_light += [comp + bessel]
Phis = [np.asmatrix(scipy.linalg.circulant(p)) for p in p_light]

# Solve convex relaxation
z = Variable(n)
Z = Symmetric(n)   # Z = zz^T
I_high = Variable()
I_low = Variable()

obj = I_high - I_low
H_mat = bmat([[np.array([[1]]), z.T], [z, Z]])   # H = [[1, z^T], [z, Z]]
I = sum([intensity(z, Z, Phi) for Phi in Phis])
cons = [diag(Z) == 1, H_mat >> 0, I[S_vec == 1] >= I_high, I[S_vec == 0] <= I_low, I_high >= I_low]
prob = Problem(Maximize(obj), cons)
prob.solve("MOSEK")
print("Status: {}".format(prob.status))

# Real mask must be binary
z_bin = np.sign(z.value)
# z_bin = rand_sign(Z.value)
m_bin = (z_bin + 1)/2
M_bin = m_bin.dot(m_bin.T)
I_bin = sum([np.diag(Phi.H.dot(M_bin).dot(Phi)) for Phi in Phis])

# Retrieve solution
I_sol = np.real(I_bin)
S_sol = np.reshape(I_sol, (k,l))   # Reshape into 2-D image
print("MSE in Substrate: {}".format(np.linalg.norm(S - S_sol)**2/n))

# Display resulting substrate
f, axarr = plt.subplots(1,2)
axarr[0].imshow(S, cmap = "Greys", interpolation = "nearest")
axarr[0].set_title("Desired Substrate")
axarr[1].imshow(S_sol, cmap = "Greys", interpolation = "nearest")
axarr[1].set_title("Solution Substrate")
plt.show()
