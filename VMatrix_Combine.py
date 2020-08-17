# %% [markdown]
# # V-Matrix Density Ratio Approach

#%%
import numpy as np
from scipy.interpolate import interp1d
# %%
from sklearn.externals.joblib import Memory

mem = Memory(location='.', verbose=0)

# %%
mem.clear()

# %%
import sklearn

cached_rbf_kernel = mem.cache(sklearn.metrics.pairwise.rbf_kernel)


class rbf_krnl(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        return cached_rbf_kernel(X, Y, gamma=self.gamma)

    def __repr__(self):
        return "RBF Gaussian gamma: " + str(self.gamma)


cached_polynomial_kernel = mem.cache(sklearn.metrics.pairwise.polynomial_kernel)


class poly_krnl(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        return cached_polynomial_kernel(X, Y, gamma=self.gamma, coef0=1)

    def __repr__(self):
        return "Polynomial deg 3 kernel gamma: " + str(self.gamma)


class poly_krnl_2(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        return cached_polynomial_kernel(X, Y, degree=2, gamma=self.gamma,
                                        coef0=1e-6)  # I use a homogeneous polynomial kernel

    def __repr__(self):
        return "Polynomial deg 2 kernel gamma: " + str(self.gamma)


class poly_krnl_inv(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        return cached_polynomial_kernel(X, Y, degree=-1, gamma=self.gamma,
                                        coef0=0.5)

    def __repr__(self):
        return "Polynomial deg -1 kernel gamma: " + str(self.gamma)


# %%
def INK_Spline_Linear(x, y, gamma):
    x = np.atleast_2d(x) * gamma
    y = np.atleast_2d(y) * gamma
    min_v = np.min(np.stack((x, y)), axis=0)

    k_p = 1 + x * y + 0.5 * np.abs(
        x - y) * min_v * min_v + min_v * min_v * min_v / 3

    return np.prod(k_p, axis=1)


def INK_Spline_Linear_Normed(x, y, gamma):
    """Computes the Linear INK-Spline Kernel
    x,y: 2-d arrays, n samples by p features
    Returns: """
    return INK_Spline_Linear(x, y, gamma) / np.sqrt(
        INK_Spline_Linear(x, x, gamma) * INK_Spline_Linear(y, y, gamma))


from sklearn.metrics import pairwise_kernels


class ink_lin_krnl(object):
    """
    Linear INK-Spline Kernel
    Assumes that the domain is [0,+inf]"""

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        idxs = np.mgrid[slice(0, X.shape[0]), slice(0, Y.shape[0])]
        res = INK_Spline_Linear_Normed(X[idxs[0].ravel()],
                                       Y[idxs[1].ravel()], self.gamma).reshape(
            X.shape[0], Y.shape[0])
        return res

    def __repr__(self):
        return "Linear INK-spline kernel (on [0,1])"


# %%

# %%
def v_mat_star_eye(X, X_prime, dummy):
    return np.eye(X.shape[0])


# %%
def ecdf(x):
    v, c = np.unique(x, return_counts='true')
    q = np.cumsum(c) / np.sum(c)
    return v, q
# %%
from numba import jit, prange, njit


# %% [markdown]
# In "V-Matrix Method of Solving Statistical Inference Problems" (Vapnik and Izmailov), the V matrix is expressed as:
#
# $$
# V_{i,j} = \prod_{k=1}^d \int \theta(x^{(k)}-X_i^{(k)})\,\theta(x^{(k)}-X_j^{(k)}) \sigma_k(x^{(k)}) d\mu(x^{(k)})
# $$
#
#
# If $\sigma(x^{(k)}) = 1$ and $d\mu(x^{(k)}) = \prod_{k=1}^d dF_\ell(x^{(k)})$ 
# $$
# V_{i,j} = \prod_{k=1}^d \nu\left(X^{(k)} > \max\left\lbrace X_i^{(k)},X_j^{(k)}\right\rbrace\right)
# $$
#
# However, the following is recommended for density ratio estimation
#
# $$
# \sigma(x_k) = \frac{1}{F_{num}(x_k)(1-F_{num}(x_k))+\epsilon}
# $$
#
# It's not clear to me why we'd be looking only at the ECDF of the numerator. Why not all the data?
#
# In any case, how do we calculate the $V_{i,j}$?

# %% [markdown]
# I would say that the integral can be approximated with a sum:
#
# $$
# \frac{1}{\ell}\sum_{x_k > \left\lbrace X_i^{(k)},X_j^{(k)}\right\rbrace} \sigma(x_k)
# $$
#
# where the $x_k$ are taken from all the data (??)

# %%
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_sigma_eye(X, X_prime, data):
    data_sorted = np.sort(data, axis=0)
    data_l = data.shape[0]

    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))
    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            acc = 1
            for k in range(X.shape[1]):
                # Let's compute the frequency of data with values larger than those for X_i and X^'_j
                f = (data_l - np.searchsorted(data_sorted[:, k],
                                              max(X[i, k], X_prime[j, k]),
                                              side="right")) / data_l
                acc *= f
            v[i, j] = acc
    return v


# %%
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_max(X, X_prime, dummy):
    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))
    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            acc = 1
            for k in range(X.shape[1]):
                acc *= 1 - max(X[i, k], X_prime[j, k])
            v[i, j] = acc
    return v


# %%

# %%
# This takes forever... 


# @mem.cache
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_sigma_ratio(X, X_prime, data):
    data_sorted = np.sort(data, 0)
    data_l = data.shape[0]
    eps = 1 / (data_l * data_l)  # Just an idea...
    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))

    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            accu = 1
            for k in range(X.shape[1]):
                dd = data_sorted[:, k]
                s = 0
                for l in data[:, k]:
                    if l > X[i, k] and l > X_prime[j, k]:
                        f = (np.searchsorted(dd, l, side="right")) / data_l
                        s += 1 / (f * (1 - f) + eps)
                accu *= (s / data_l)
            v[i, j] = accu
    return v


# %% [markdown]
# ### Experimental V-matrices

# %%
from statsmodels.distributions.empirical_distribution import ECDF


# %%
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_star_sigma_ratio_approx(X, X_prime, data):
    data_sorted = np.sort(data, 0)
    data_l = data.shape[0]
    eps = 1 / (data_l)  # Just an idea...
    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))

    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            accu = 1
            for k in range(X.shape[1]):
                dd = data_sorted[:, k]
                f = (data_l - np.searchsorted(dd, np.maximum(X[i, k],
                                                             X_prime[j, k]),
                                              side="right")) / data_l
                f1 = (np.searchsorted(dd, X[i, k], side="right")) / data_l
                f2 = (np.searchsorted(dd, X_prime[j, k], side="right")) / data_l
                accu *= f / (f1 * f2 * (1 - f2) * (1 - f1) + eps)
            v[i, j] = accu
    return v / np.max(v)


# %%
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_star_sigma_oneside_approx(X, X_prime, data):
    data_sorted = np.sort(data, 0)
    data_l = data.shape[0]
    eps = 1 / (data_l)  # Just an idea...
    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))

    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            accu = 1
            for k in range(X.shape[1]):
                dd = data_sorted[:, k]
                f = (data_l - np.searchsorted(dd, np.maximum(X[i, k],
                                                             X_prime[j, k]),
                                              side="right")) / data_l
                f1 = (np.searchsorted(dd, X[i, k], side="right")) / data_l
                f2 = (np.searchsorted(dd, X_prime[j, k], side="right")) / data_l
                accu *= f / ((1 - f1) * (1 - f2) + eps)
            v[i, j] = accu
    return v / np.max(v)


# %%
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_star_sigma_rev_approx(X, X_prime, data):
    data_sorted = np.sort(data, 0)
    data_l = data.shape[0]
    eps = 1 / (data_l)  # Just an idea...
    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))

    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            accu = 1
            for k in range(X.shape[1]):
                dd = data_sorted[:, k]
                f = (data_l - np.searchsorted(dd, np.maximum(X[i, k],
                                                             X_prime[j, k]),
                                              side="right")) / data_l
                f1 = (np.searchsorted(dd, X[i, k], side="right")) / data_l
                f2 = (np.searchsorted(dd, X_prime[j, k], side="right")) / data_l
                accu *= f * f1 * (1 - f1) * f2 * (1 - f2)
            v[i, j] = accu
    return v


# %%
@mem.cache
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_star_sigma_oneside(X, X_prime, X_num):
    X_num_sorted = np.sort(X_num, 0)
    X_num_l = X_num.shape[0]
    eps = 1e-6
    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))
    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            acc = 1
            for k in range(X.shape[1]):
                f = np.searchsorted(X_num_sorted[:, k],
                                    np.maximum(X[i, k], X_prime[j, k]),
                                    side="right") / X_num_l
                acc *= 1 / (1 + f)
                # acc *= 1/(f*(1-f)+eps)
            v[i, j] = acc
    return v / np.max(v)


# %%
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])')
def v_mat_star_sigma_log(X, X_prime, dummy):
    v = np.zeros(shape=(X.shape[0], X_prime.shape[0]))
    for i in prange(X.shape[0]):
        for j in range(X_prime.shape[0]):
            acc = 1
            for k in range(X.shape[1]):
                acc *= -np.log(np.maximum(X[i, k], X_prime[j, k]))
            v[i, j] = acc
    return v / np.max(v)


# %%
import cvxopt


def DensityRatio_QP(X_den, X_num, kernel, g, v_matrix, ridge=1e-3):
    """
    The function computes a model of the density ratio.
    The function is in the form $A^T K$
    The function returns the coefficients $\alpha_i$ and the bias term b
    """
    l_den, d = X_den.shape
    l_num, d_num = X_num.shape

    # TODO: Check d==d_num

    ones_num = np.matrix(np.ones(shape=(l_num, 1)))
    zeros_den = np.matrix(np.zeros(shape=(l_den, 1)))

    gram = kernel(X_den)
    K = np.matrix(gram + ridge * np.eye(l_den))
    # K = np.matrix(gram)   # No ridge

    print("K max, min: %e, %e" % (np.max(K), np.min(K)))

    data = np.concatenate((X_den, X_num))
    if callable(v_matrix):
        V = np.matrix(v_matrix(X_den, X_den, data))
        V_star = np.matrix(v_matrix(X_den, X_num, data))  # l_den by l_num
    else:
        return -1

    print("V max,min: %e, %e" % (np.max(V), np.min(V)))
    print("V_star max,min: %e, %e" % (np.max(V_star), np.min(V_star)))

    tgt1 = K * V * K
    print("K*V*K max, min: %e, %e" % (np.max(tgt1), np.min(tgt1)))

    tgt2 = g * K
    print("g*K max, min: %e, %e" % (np.max(tgt2), np.min(tgt2)))

    P = cvxopt.matrix(2 * (tgt1 + tgt2))

    q_ = -2 * (l_den / l_num) * (K * V_star * ones_num)

    print("q max, min: %e, %e" % (np.max(q_), np.min(q_)))
    q = cvxopt.matrix(q_)

    #### Let's construct the inequality constraints

    # Now create G and h
    G = cvxopt.matrix(-K)
    h = cvxopt.matrix(zeros_den)
    # G = cvxopt.matrix(np.vstack((-K,-np.eye(l_den))))
    # h = cvxopt.matrix(np.vstack((zeros_den,zeros_den)))

    # Let's construct the equality constraints

    A = cvxopt.matrix((1 / l_den) * K * V_star * ones_num).T
    b = cvxopt.matrix(np.ones(1))

    return cvxopt.solvers.qp(P, q, G, h, A, b, options=dict(
        maxiters=50))  #### For expediency, we limit the number of iterations


# %%
def RKHS_Eval(A, X_test, X_train, kernel, c=0):
    gramTest = kernel(X_test, X_train)

    return np.dot(gramTest, A) + c


# %%
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.preprocessing import StandardScaler


class DensityRatio_Estimator(BaseEstimator, RegressorMixin):
    """Custom Regressor for density ratio estimation"""

    def __init__(self, krnl=rbf_krnl(1), g=1, v_matrix=v_mat_sigma_eye):
        self.krnl = krnl
        self.g = g
        self.v_matrix = v_matrix

    def fit(self, X_den, X_num):

        self.X_train_ = np.copy(X_den)

        res = DensityRatio_QP(self.X_train_,
                              X_num,
                              kernel=self.krnl,
                              g=self.g,
                              v_matrix=self.v_matrix)
        self.A_ = res['x']
        return self

    def predict(self, X):
        if self.A_ is None:
            return None  # I should raise an exception

        return self.predict_proba(X)

    def predict_proba(self, X):
        if self.A_ is None:
            return None  # I should raise an exception

        pred = RKHS_Eval(A=self.A_,
                         X_test=X,
                         X_train=self.X_train_,
                         kernel=self.krnl)
        return np.clip(pred, a_min=0, a_max=None, out=pred)


# %%

# %%
def NeymanPearson_VMatrix(p_a, p_b, h0, p_a_test, p_b_test, g=0.5,
                          krnl=ink_lin_krnl(1), v_matrix=v_mat_sigma_ratio,
                          diag=False):
    p_h0 = np.hstack((p_a[h0].reshape(-1, 1), p_b[h0].reshape(-1, 1)))
    p_h1 = np.hstack((p_a[~h0].reshape(-1, 1), p_b[~h0].reshape(-1, 1)))

    dre = DensityRatio_Estimator(v_matrix=v_matrix,
                                 krnl=krnl,
                                 g=g)
    dre.fit(X_den=p_h1, X_num=p_h0)

    lmbd_h0 = np.clip(dre.predict(p_h0), 1e-10, None)
    v, q = ecdf(lmbd_h0)
    v = np.concatenate(([0],
                        v))  # This may be OK for this application but it is not correct in general
    q = np.concatenate(([0], q))
    NP_calibr = interp1d(v, q, bounds_error=False, fill_value="extrapolate")

    p_test = np.hstack((p_a_test.reshape(-1, 1), p_b_test.reshape(-1, 1)))
    p_npcomb = NP_calibr(dre.predict(p_test))

    if diag:
        f, axs = plt.subplots(1, 3, figsize=(12, 4))
        x = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, x)
        lambd_grid = dre.predict(
            np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))))
        comb_p_grid = NP_calibr(lambd_grid)
        # im = axs[0].imshow(lambd_grid.reshape(100,100),interpolation=None,origin='lower')   
        im = axs[0].contourf(xx, yy, lambd_grid.reshape(100, 100));
        f.colorbar(im, ax=axs[0])
        axs[0].set_title("Lambda")

        # im = axs[2].imshow(comb_p_grid.reshape(100,100),interpolation=None,origin='lower')
        im = axs[2].contourf(xx, yy, comb_p_grid.reshape(100, 100), vmin=0,
                             vmax=1);
        f.colorbar(im, ax=axs[2])
        axs[2].set_title("Combined p-value")

        x = np.linspace(0, np.max(lmbd_h0), 100)
        axs[1].plot(x, NP_calibr(x))
        axs[1].set_title("Lambda to p-value")
        print("Max:", np.max(comb_p_grid))
        print("Min:", np.min(comb_p_grid))

        f.tight_layout()

    return np.clip(p_npcomb.ravel(), 0, 1)



# %%
# %%time
@mem.cache
def comb_np_vm(ps, ps_cal, h0_cal):
#    nnp_kwargs = dict(g=1e-5, krnl=rbf_krnl(4.5),
    nnp_kwargs = dict(g=1e-2, krnl=rbf_krnl(1.5),
#                 v_matrix=v_mat_star_sigma_rev_approx)
                 v_matrix=v_mat_star_eye)

 
    return NeymanPearson_VMatrix(ps_cal[:,0], ps_cal[:,1], h0_cal, ps[:,0], ps[:,1], **nnp_kwargs)