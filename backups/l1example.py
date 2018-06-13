import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
p_num = 5

S = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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


# I_high = 1155.5201554916755
# I_low = 955.5201554916755

I_high = 1319.293853597313
I_low = 819.293853597313

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

obj = sum(neg(I[S_vec == 1] - I_high)**2) + sum(pos(I[S_vec == 0] - I_low)**2) + norm(Z,1)

cons = [diag(Z) == 1, H_mat >> 0]

prob = Problem(Minimize(obj), cons)
prob.solve("MOSEK")
print("Status: {}".format(prob.status))

#print("rank of matrix Z is", np.linalg.matrix_rank(Z.value))

I = sum([intensity(z.value, Z.value, Phi) for Phi in Phis])
I = I.value
for i in range(len(I)):
    if I[i] <= I_low:
        I[i] = 0
    elif I[i] >= I_high:
        I[i] = 1

S_sol = np.reshape(I, (k,l))   # Reshape into 2-D image
print("MSE in Substrate: {}".format(np.linalg.norm(S - S_sol)**2/n))

# Display resulting substrate
f, axarr = plt.subplots(1,2)
axarr[0].imshow(S, cmap = "Greys", interpolation = "nearest")
axarr[0].set_title("Desired Substrate")
axarr[1].imshow(S_sol, cmap = "Greys", interpolation = "nearest")
axarr[1].set_title("CVX Solution Substrate")
plt.show()


w, v = np.linalg.eig(Z.value)
idx = np.argmax(w)
what = np.zeros_like(w)
what[idx] = w[idx]

zhat = v.dot(what)

zhat = zhat.reshape((n,1))
np.shape(zhat)
Zhat = zhat.dot(zhat.T)

I = sum([intensity(zhat, Zhat, Phi) for Phi in Phis])
I = I.value

for i in range(len(I)):
    if I[i] <= I_low:
        I[i] = 0
    elif I[i] >= I_high:
        I[i] = 255
    else:
    	I[i] = 125

S_sol = np.reshape(I, (k,l))   # Reshape into 2-D image
print("MSE in Substrate: {}".format(np.linalg.norm(S - S_sol)**2/n))

# Display resulting substrate
f, axarr = plt.subplots(1,2)
axarr[0].imshow(S, cmap = "Greys", interpolation = "nearest")
axarr[0].set_title("Desired Substrate")
axarr[1].imshow(S_sol, cmap = "Greys", interpolation = "nearest")
axarr[1].set_title("Solution Substrate")
plt.show()