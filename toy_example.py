import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
m = 5
# n = 50
I_high = 0.75
I_low = 0.25
# S = np.random.randint(0, 2, size = (5,5))
S = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
			  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
k,l = S.shape
n = np.prod(S.shape)
S_vec = S.flatten()   # Flatten into vector

# Generate light wavelengths
z = np.arange(n)
p_light = []
for i in range(m):
	data = np.random.randn(n,2)
	comp = np.apply_along_axis(lambda args: [complex(*args)], 1, data)
	bessel = scipy.special.jv(1, z).reshape((n,1))
	# p_light += [comp]
	p_light += [comp + bessel]
Phis = [np.asmatrix(scipy.linalg.circulant(p)) for p in p_light]
# Phis_split = [(np.real(Phi), np.imag(Phi)) for Phi in Phis]

# Solve convex relaxation
M = Semidef(n)
obj = trace(M)
I = sum([diag(Phi.H*M*Phi) for Phi in Phis])
# I = sum([Preal.T*M*Preal + Pimag.T*M*Pimag for Preal, Pimag in Phis_split])
cons = [I[S_vec == 1] >= I_high, I[S_vec == 0] <= I_low]
prob = Problem(Minimize(obj), cons)
prob.solve("MOSEK")

# Real mask must be binary
M_bin = np.sign(M.value)
I_bin = sum([np.diag(Phi.H.dot(M_bin).dot(Phi)) for Phi in Phis])

# Retrieve solution
# I_sol = np.real(I.value)
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
