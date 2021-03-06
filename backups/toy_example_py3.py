import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
p_num = 5
# S = np.random.randint(0, 2, size = (5,5))
S = np.loadtxt(open("data/smiley_array.txt", "rb"), delimiter=",")

k,l = S.shape
n = np.prod(S.shape)
S_vec = S.flatten()   # Flatten into vector

# Generate light wavelengths
v = np.arange(n)
p_light = []
for i in range(p_num):
	data = np.random.randn(n,2)
	comp = np.apply_along_axis(lambda args: [complex(*args)], 1, data)
	bessel = scipy.special.jv(1,v).reshape((n,1))
	p_light += [comp + bessel]
Phis = [np.asmatrix(scipy.linalg.circulant(p)) for p in p_light]

# Normalized light intensity
def intensity(z, Z, Phi):
	ones = np.ones((n,1))
	term1 = diag(real(Phi.H*Z*Phi))
	term2 = np.diag(np.real(Phi.H.dot(ones).dot(ones.T).dot(Phi)))
	term3 = 2*diag(real(Phi.H*z*ones.T*Phi))
	return (term1 + term2 + term3)/4

# Solve convex relaxation
z = Variable((n,1))
Z = Variable((n,n), symmetric = True)   # Z = zz^T
I_high = Variable()
I_low = Variable()

obj = I_high - I_low
H_mat = bmat([[np.array([[1]]), z.T], [z, Z]])   # H = [[1, z^T], [z, Z]]
I = sum([intensity(z, Z, Phi) for Phi in Phis])
cons = [diag(Z) == 1, H_mat >> 0, I[S_vec == 1] >= I_high, I[S_vec == 0] <= I_low, I_high >= I_low]
prob = Problem(Maximize(obj), cons)
prob.solve("SCS")
print("Status: {}".format(prob.status))

# Real mask must be binary
z_bin = np.sign(z.value)
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
