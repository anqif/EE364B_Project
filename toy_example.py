import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
m = 5
n = 10
I_high = 0.75
I_low = 0.25
# S = np.random.randint(0, 2, size = (n,n))
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
expr = sum([Phi.H*M*Phi for Phi in Phis])
# expr = sum([Preal.T*M*Preal + Pimag.T*M*Pimag for Preal, Pimag in Phis_split])
cons = [expr[S == 1] >= I_high, expr[S == 0] <= I_low]
prob = Problem(Minimize(obj), cons)
prob.solve()

# Display resulting mask
plt.imshow(S, cmap = "Greys", interpolation = "nearest")
plt.title("Desired Mask")
plt.show()

plt.imshow(M.value, cmap = "Greys", interpolation = "nearest")
plt.title("Solution Mask")
plt.show()
