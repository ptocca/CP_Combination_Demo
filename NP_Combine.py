# %% [markdown]
# # Neyman-Pearson

# %%
import numpy as np
from scipy.interpolate import UnivariateSpline
# %%
def ecdf(x):
    v, c = np.unique(x, return_counts='true')
    q = np.cumsum(c) / np.sum(c)
    return v, q

def splineEst(data, n_knots=20, s=0.3):
    k = np.linspace(0, 1, n_knots + 1)

    # UnivariateSpline() below requires that the x be strictly increasing
    # quantiles might be the same...

    h, bins = np.histogram(data, bins=n_knots, density=True)

    ss = UnivariateSpline(0.5 * (bins[:-1] + bins[1:]), h, k=3, s=s, ext=3)
    return ss
# %%
def nnp(p_a, p_b, h0, p_a_test, p_b_test, densityEstimator=splineEst):
    p_a_h0 = p_a[h0]

    kde = densityEstimator(p_a_h0)
    l_h0 = kde(p_a)

    p_a_h1 = p_a[~h0]

    kde = densityEstimator(p_a_h1)
    l_h1 = kde(p_a)

    lmbd_a = l_h0 / l_h1

    # lmbd_a = np.clip(lmbd_a,1e-10,1e+10)

    p_a_u, i_u = np.unique(p_a, return_index=True)
    lmbd_a_int = UnivariateSpline(p_a_u, lmbd_a[i_u], k=1, s=0, ext=3)

    ########################################################################################

    # Now compute lambda for p_b

    p_b_h0 = p_b[h0]

    kde = densityEstimator(p_b_h0)
    l_h0 = kde(p_b)

    p_b_h1 = p_b[~h0]

    kde = densityEstimator(p_b_h1)
    l_h1 = kde(p_b)

    lmbd_b = l_h0 / l_h1

    # lmbd_b = np.clip(lmbd_a,1e-10,1e+10)
    p_b_u, i_u = np.unique(p_b, return_index=True)
    lmbd_b_int = UnivariateSpline(p_b_u, lmbd_b[i_u], k=1, s=0, ext=3)

    ######################################################################
    # Combine the lambdas assuming independence
    # lmbd_comb = lmbd_a_interp(p_a)*lmbd_b_interp(p_b)
    lmbd_comb = lmbd_a * lmbd_b

    # lmbd_comb_interp = UnivariateSpline(eval_points,lmbd_comb,k=1,s=0,ext=3)

    v, q = ecdf(lmbd_comb[h0])

    NP_calibr = UnivariateSpline(v, q, k=1, s=0, ext=3)

    # This can take a while
    p_npcomb = NP_calibr(lmbd_a_int(p_a_test) * lmbd_b_int(p_b_test))

    return p_npcomb


# %%
# I really should write this in the main script
# The comb_method() takes as parameters:
# the p-values of the test objects as a Nx2 array
# the p-values of the calibration examples
# a mask that selects the H0 calibration example, i.e. those with the same label as for the p-values in hand 

# In Neyman-Pearson we also use the p-values of the combination calibration examples with the OPPOSITE label as for the p-value
def comb_nnp(ps, ps_cal, h0_cal):
    return nnp(ps_cal[:,0], ps_cal[:,1], h0_cal, ps[:,0], ps[:,1])


