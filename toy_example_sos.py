import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(1)
p_num = 5
# w_high = 0.75   # Circuit array: MSE = 0.07
# w_high = 0.5   # Smiley: MSE = 0.02
# lam = 0.5

# Rectangle:
w_high = 1.0   # Small block
I_high = 1155.5201554916755
I_low = 955.5201554916755

# I_high = Variable()
# I_low = Variable()

save_figs = True
in_name = "data/small_block_array.txt"
out_name_d = "figures/small_block_desired.png"
out_name_r = "figures/small_block_result.png"
out_name_rc = "figures/small_block_result_cut.png"
# S = np.random.randint(0, 2, size = (5,5))
S = np.loadtxt(open(in_name, "rb"), delimiter=",")

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

# Reduce to rank 1 matrix decomposition
def reduce_rank(Z):
	m, n = Z.shape
	w, v = np.linalg.eigh(Z)
	w_hat = np.sqrt(w[-1])   # Take largest singular value
	z = w_hat*v[:,-1]
	return z

# Solve convex relaxation
z = Variable((n,1))
Z = Variable((n,n), symmetric = True)   # Z = zz^T

H_mat = bmat([[np.array([[1]]), z.T], [z, Z]])   # H = [[1, z^T], [z, Z]]
I = sum([intensity(z, Z, Phi) for Phi in Phis])
# reg = lam*norm(Z, "nuc")
obj = w_high*sum(neg(I[S_vec == 1] - I_high)) + sum(pos(I[S_vec == 0] - I_low))
cons = [diag(Z) == 1, H_mat >> 0]
# cons = [diag(Z) == 1, H_mat >> 0, I_high >= I_low, I_low >= 0]
prob = Problem(Minimize(obj), cons)
prob.solve("MOSEK")
print("Status: {}".format(prob.status))
# print("Rank of Z: {}".format(np.linalg.matrix_rank(Z.value)))
print("Eigenvalues of Z: {}".format(np.linalg.eigvals(Z.value)))
# print("I_high: {}".format(I_high.value))
# print("I_low: {}".format(I_low.value))

# Real mask must be binary
z_bin = np.sign(z.value)
# z_bin = np.sign(reduce_rank(Z.value))
m_bin = (z_bin + 1)/2
M_bin = m_bin.dot(m_bin.T)
I_bin = sum([np.diag(Phi.H.dot(M_bin).dot(Phi)) for Phi in Phis])

# Retrieve and threshold solution
I_sol = np.real(I_bin)
S_sol = np.reshape(I_sol, (k,l))   # Reshape into 2-D image
I_sol_cut = np.full(I_sol.shape, 0.5)
I_sol_cut[I_sol >= I_high] = 1
I_sol_cut[I_sol <= I_low] = 0
# I_sol_cut[I_sol >= I_high.value] = 1
# I_sol_cut[I_sol <= I_low.value] = 0
S_sol_cut = np.reshape(I_sol_cut, (k,l))   # Reshape into 2-D image
# print("MSE in Substrate: {}".format(np.linalg.norm(S - S_sol)**2/n))
print("MSE in Thresholded Substrate: {}".format(np.linalg.norm(S - S_sol_cut)**2/n))

# Display resulting substrate
f, axarr = plt.subplots(1,3)
axarr[0].imshow(S, cmap = "Greys", interpolation = "nearest")
axarr[0].set_title("Desired Substrate")
axarr[1].imshow(S_sol, cmap = "Greys", interpolation = "nearest")
axarr[1].set_title("Solution Substrate")
axarr[2].imshow(S_sol_cut, cmap = "Greys", interpolation = "nearest")
axarr[2].set_title("Thresholded Substrate")
plt.show()

# Save images for paper
if save_figs:
	f = plt.figure()
	plt.imshow(S, cmap = "Greys", interpolation = "nearest")
	plt.axis("off")
	f.savefig(out_name_d, dpi = f.dpi)

	f = plt.figure()
	plt.imshow(S_sol, cmap = "Greys", interpolation = "nearest")
	plt.axis("off")
	f.savefig(out_name_r, dpi = f.dpi)
	
	f = plt.figure()
	plt.imshow(S_sol_cut, cmap = "Greys", interpolation = "nearest")
	plt.axis("off")
	f.savefig(out_name_rc, dpi = f.dpi)
