import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
p_num = 5

S = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
			  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			  [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

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

#I_high = 1319.293853597313
#I_low = 819.293853597313

I_high =  1120.0282185006422
I_low = 620.0282185006423

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


H_mat = bmat([[np.array([[1]]), z.T], [z, Z]])   # H = [[1, z^T], [z, Z]]
I = sum([intensity(z, Z, Phi) for Phi in Phis])

obj = 0.75*sum(neg(I[S_vec == 1] - I_high)) + sum(pos(I[S_vec == 0] - I_low))

cons = [diag(Z) == 1, H_mat >> 0]

prob = Problem(Minimize(obj), cons)
prob.solve("MOSEK")
print("Status: {}".format(prob.status))

z_bin = np.sign(z.value)
# z_bin = np.sign(reduce_rank(Z.value))
m_bin = (z_bin + 1)/2
M_bin = m_bin.dot(m_bin.T)
I_bin = sum([np.diag(Phi.H.dot(M_bin).dot(Phi)) for Phi in Phis])

# Retrieve and threshold solution
I_sol = np.real(I_bin)
S_sol = np.reshape(I_sol, (k,l))   # Reshape into 2-D image

# desired result
fig = plt.figure()
plt.imshow(S, cmap = "Greys", interpolation = "nearest")
plt.show()

# threshold
for i in range(len(I_sol)):
    if I_sol[i] >= I_high:
        I_sol[i] = 255
    elif I_sol[i] <= I_low:
        I_sol[i] = 0
    else:
        I_sol[i] = 125
        
S_sol = np.reshape(I_sol, (k,l))

## thresholded result
fig = plt.figure()
plt.imshow(S_sol, cmap = "Greys", interpolation = "nearest")
plt.show()

out_name_rc = "figures/circuit_cut.png"
fig.savefig(out_name_rc, dpi = fig.dpi)

