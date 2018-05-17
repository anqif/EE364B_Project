import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
m = 5
n = 25
I_high = 0.75
I_low = 0.25
S = np.random.randint(0, 2, size = (n,1))
# S = np.random.randint(0, 2, size = (n,n))
# S = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#			  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
#			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#			  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
#			  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
#			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Generate light wavelengths
p_light = []
for i in range(m):
	data = np.random.randn(n,2)
	comp = np.apply_along_axis(lambda args: [complex(*args)], 1, data)
	p_light += [comp]
Phis = [np.asmatrix(scipy.linalg.circulant(p)) for p in p_light]
# Phis_split = [(np.real(Phi), np.imag(Phi)) for Phi in Phis]

# Solve convex relaxation
M = Semidef(n)
obj = trace(M)
I = sum([diag(Phi.H*M*Phi) for Phi in Phis])
# I = sum([Phi.H*M*Phi for Phi in Phis])
# I = sum([Preal.T*M*Preal + Pimag.T*M*Pimag for Preal, Pimag in Phis_split])
cons = [I[S == 1] >= I_high, I[S == 0] <= I_low]
prob = Problem(Minimize(obj), cons)
prob.solve()
I_sol = np.real(I.value)
print("MSE in Substrate: {}".format(np.linalg.norm(S - I_sol)**2/n))

# Display resulting mask
f, axarr = plt.subplots(1,2)
axarr[0].imshow(S, cmap = "Greys", interpolation = "nearest")
axarr[0].set_title("Desired Substrate")
axarr[1].imshow(I_sol, cmap = "Greys", interpolation = "nearest")
axarr[1].set_title("Solution Substrate")
plt.show()
