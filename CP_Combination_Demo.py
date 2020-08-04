# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python(comb_demo)
#     language: python
#     name: comb_demo
# ---

# %% [markdown]
# # CP combination demo

# %%
import numpy as np
import scipy.stats as ss
from sklearn.model_selection import train_test_split
import pandas as pd

# %%
from IPython.display import display, HTML

display(HTML("<style>.container {width: 90% !important;} </style>"))

# %%
import panel as pn

pn.extension(comm='ipywidgets')

# %%
# %matplotlib inline
from matplotlib.figure import Figure

# %%
import param

# %%
from CP import *


# %%
def plot_score_hist(alpha_0_a, alpha_0_b, alpha_1_a, alpha_1_b):
    f = Figure(figsize=(18, 6))
    ax_a = f.add_subplot(1, 3, 1)
    ax_a.hist([alpha_0_a, alpha_1_a], bins=np.linspace(-10, 10, 51), label=("Negative","Positive"), color=("g","r"))
    ax_a.set_title("Classifier A")
    ax_a.legend()
    ax_a.set_xlabel('"Score"')

    ax_b = f.add_subplot(1, 3, 2, sharey=ax_a)
    ax_b.hist([alpha_0_b, alpha_1_b], bins=np.linspace(-10, 10, 51), label=("Negative","Positive"), color=("g","r"))
    ax_b.set_title("Classifier B")
    ax_b.legend()
    ax_b.set_xlabel('"Score"')
 
    ax_c = f.add_subplot(1, 3, 3)
    ax_c.plot(alpha_0_a, alpha_0_b, "g.", alpha=0.05, label="Negative");
    ax_c.plot(alpha_1_a, alpha_1_b, "r.", alpha=0.05, label="Positive");
    ax_c.set_xlabel("Classifier A")
    ax_c.set_ylabel("Classifier B")
    ax_c.legend()
    f.suptitle("Histograms of simulated scores\n",
               fontsize=16);
    return f


class SynthDataSet(param.Parameterized):
    N = param.Integer(default=2000, bounds=(100, 10000))
    percentage_of_positives = param.Number(default=50.0, bounds=(0.1, 100.0))
    seed = param.Integer(default=0, bounds=(0, 32767))
    cc = param.Number(default=0.0, bounds=(-1.0, 1.0))
    var = param.Number(default=1.0, bounds=(0.5, 2.0))

    micp_calibration_fraction = param.Number(default=0.5, bounds=(0.01, 0.5))
    comb_calibration_fraction = param.Number(default=0.3, bounds=(0.01, 0.5))

    # Outputs
    output = param.Dict(default=dict(),
                        precedence=-1)  # To have all updates in one go

    n = 2

    def __init__(self, **params):
        super(SynthDataSet, self).__init__(**params)
        self.update()

    def update(self):
        output = dict()

        cov = self.cc * np.ones(shape=(self.n, self.n))
        cov[np.diag_indices(self.n)] = self.var

        np.random.seed(self.seed)

        positives_number = int(self.N * self.percentage_of_positives / 100)
        negatives_number = self.N - positives_number
        
        try:
            alpha_neg = ss.multivariate_normal(mean=[-1, -1], cov=cov).rvs(
                size=(negatives_number,))
            alpha_pos = ss.multivariate_normal(mean=[1, 1], cov=cov).rvs(
                size=(positives_number,))
        except:
            placeholder = np.array([0.0])
            output['scores_cal_a'] = placeholder
            output['scores_pcal_a'] = placeholder
            output['scores_cal_b'] = placeholder
            output['scores_pcal_b'] = placeholder
            output['y_cal'] = placeholder
            output['y_pcal'] = placeholder
            output['scores_test_a'] = placeholder
            output['scores_test_b'] = placeholder
            output['y_test'] = placeholder
            self.output = output
            return
            

        alpha_neg_a = alpha_neg[:, 0]
        alpha_neg_b = alpha_neg[:, 1]
        alpha_pos_a = alpha_pos[:, 0]
        alpha_pos_b = alpha_pos[:, 1]

        scores_a = np.concatenate((alpha_neg_a, alpha_pos_a))
        scores_b = np.concatenate((alpha_neg_b, alpha_pos_b))
        y = np.concatenate((np.zeros(negatives_number, dtype=np.int8),
                            np.ones(positives_number, dtype=np.int8)))

        micp_calibration_size = int(self.micp_calibration_fraction * self.N)
        comb_calibration_size = int(self.comb_calibration_fraction * self.N)
        scores_tr_a, output['scores_test_a'], \
        scores_tr_b, output['scores_test_b'], \
        y_tr, output['y_test'] = train_test_split(scores_a, scores_b, y,
                                                  train_size=micp_calibration_size + comb_calibration_size,
                                                  stratify=y)

        output['scores_cal_a'], output['scores_pcal_a'], \
        output['scores_cal_b'], output['scores_pcal_b'], \
        output['y_cal'], output['y_pcal'] = train_test_split(scores_tr_a,
                                                             scores_tr_b, y_tr,
                                                             train_size=micp_calibration_size,
                                                             stratify=y_tr)

        self.output = output

    @pn.depends("N", "percentage_of_positives", "seed", "cc", "var",
                "micp_calibration_fraction", "comb_calibration_fraction")
    def view(self):
        self.update()
        f = plot_score_hist(
            self.output['scores_cal_a'][self.output['y_cal'] == 0],
            self.output['scores_cal_b'][self.output['y_cal'] == 0],
            self.output['scores_cal_a'][self.output['y_cal'] == 1],
            self.output['scores_cal_b'][self.output['y_cal'] == 1])

        return f

    def view2(self):
        return "# %d" % self.N


sd = SynthDataSet()


# %%
def p_plane_plot(p_0, p_1, y, title_part, pics_title_part):
    def alpha_y(y):
        """Tune the transparency"""
        a = 1 - np.sum(y) / 10000
        if a < 0.05:
            a = 0.05
        return a

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.plot(p_0[y == 0], p_1[y == 0], label='Negative',
             alpha=alpha_y(y == 0));
    ax2.plot(p_0[y == 1], p_1[y == 1], label='Positive',
             alpha=alpha_y(y == 1));
    ax1.set_title("Inactives $(p_0,p_1)$ for " + title_part, fontsize=16)
    ax2.set_title("Actives $(p_0,p_1)$ for " + title_part, fontsize=16)
    ax1.set_xlabel('$p_0$', fontsize=14)
    ax1.set_ylabel('$p_1$', fontsize=14)

    ax2.set_xlabel('$p_0$', fontsize=14)

    ax1.grid()
    ax2.grid()


# %%
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, \
    interp1d


def ecdf(x):
    v, c = np.unique(x, return_counts='true')
    q = np.cumsum(c) / np.sum(c)
    return v, q


def ECDF_cal_p(p_test, p_cal):
    v, q = ecdf(p_cal)
    v = np.concatenate(([0], v))
    q = np.concatenate(([0], q))
    us = interp1d(v, q, bounds_error=False, fill_value=(0, 1))
    return us(p_test)


# %% [markdown]
# # Let's apply MICP as usual

# %% [markdown]
# Let's now redo things computing $p_0$ and $p_1$.
#
# Let's assume that the 'alphas' are actually the decision function values out of an SVM.

# %%
from CP import pValues


# %%
def pValue_hist(p_0, p_1, y, pic_title=None,
                labels_names=["Negative", "Positive"]):
    u_l = unique_labels(y)
    if len(u_l) != 2:
        return

    f = Figure(figsize=(18, 6))
    ax_0 = f.add_subplot(1, 2, 1)

    ax_0.hist((p_0[y == u_l[0]], p_0[y == u_l[1]]),
              bins=np.linspace(0, 1, 101),
              label=labels_names);
    ax_0.set_title("$p_{%s}$" % str(u_l[0]), fontsize=14)
    if not (labels_names is None):
        ax_0.legend()

    ax_1 = f.add_subplot(1, 2, 2)
    ax_1.hist((p_1[y == u_l[0]], p_1[y == u_l[1]]),
              bins=np.linspace(0, 1, 101),
              label=labels_names);
    ax_1.set_title("$p_{%s}$" % str(u_l[1]), fontsize=14)
    if not (labels_names is None):
        ax_1.legend()

    if not (pic_title is None):
        f.suptitle("p-value histograms for %s" % pic_title, fontsize=16)

    return f


# %%

def ncm(scores, label):
    if label == 1:
        return -scores
    else:
        return scores


class MICP(param.Parameterized):
    sd = param.Parameter(precedence=-1)
    p_0_a = param.Array(precedence=-1)
    p_1_a = param.Array(precedence=-1)
    p_0_b = param.Array(precedence=-1)
    p_1_b = param.Array(precedence=-1)
    p_0_a_cal = param.Array(precedence=-1)
    p_1_a_cal = param.Array(precedence=-1)
    p_0_b_cal = param.Array(precedence=-1)
    p_1_b_cal = param.Array(precedence=-1)

    def __init__(self, sd, **params):
        self.sd = sd
        super(MICP, self).__init__(**params)
        self.update()

    def aux_update_(self, scores_cal_a, scores_cal_b, scores_pcal_a,
                    scores_pcal_b, scores_test_a, scores_test_b, y_cal, y_pcal,
                    y_test):
        randomize = False

        with param.batch_watch(self):
            self.p_0_a = pValues(
                calibrationAlphas=ncm(scores_cal_a[y_cal == 0], 0),
                testAlphas=ncm(scores_test_a, 0),
                randomized=randomize)
            self.p_1_a = pValues(
                calibrationAlphas=ncm(scores_cal_a[y_cal == 1], 1),
                testAlphas=ncm(scores_test_a, 1),
                randomized=randomize)

            self.p_0_b = pValues(
                calibrationAlphas=ncm(scores_cal_b[y_cal == 0], 0),
                testAlphas=ncm(scores_test_b, 0),
                randomized=randomize)
            self.p_1_b = pValues(
                calibrationAlphas=ncm(scores_cal_b[y_cal == 1], 1),
                testAlphas=ncm(scores_test_b, 1),
                randomized=randomize)
            self.p_0_a_cal = pValues(calibrationAlphas=ncm(scores_cal_a[y_cal == 0], 0),
                                     testAlphas=ncm(scores_pcal_a, 0),
                                     randomized=randomize)
            self.p_1_a_cal = pValues(calibrationAlphas=ncm(scores_cal_a[y_cal == 1], 1),
                                     testAlphas=ncm(scores_pcal_a, 1),
                                     randomized=randomize)

            self.p_0_b_cal = pValues(calibrationAlphas=ncm(scores_cal_b[y_cal == 0], 0),
                                     testAlphas=ncm(scores_pcal_b, 0),
                                     randomized=randomize)
            self.p_1_b_cal = pValues(calibrationAlphas=ncm(scores_cal_b[y_cal == 1], 1),
                                     testAlphas=ncm(scores_pcal_b, 1),
                                     randomized=randomize)

    @pn.depends("sd.output", watch=True)
    def update(self):
        self.aux_update_(**self.sd.output)

    @pn.depends("p_0_a", "p_1_a", "p_0_b", "p_1_b")
    def view(self):
        return pn.Column(
            pValue_hist(self.p_0_a, self.p_1_a, self.sd.output['y_test']),
            pValue_hist(self.p_0_b, self.p_1_b, self.sd.output['y_test']))

    @pn.depends("p_0_a", "p_1_a", "p_0_b", "p_1_b")
    def view_tables(self):
        c_a_f = cp_cm_widget(self.p_0_a, self.p_1_a, self.sd.output['y_test'])
        c_b_f = cp_cm_widget(self.p_0_b, self.p_1_b, self.sd.output['y_test'])

        return pn.Row(c_a_f, c_b_f)

    @pn.depends("p_0_a", "p_1_a", "p_0_b", "p_1_b")
    def view_p_plane(self):
        f = Figure(figsize=(18, 6))
        ax = f.add_subplot(1, 1, 1)
        ax.plot(self.p_0_a, self.p_0_b, "g.", alpha=0.05, label="$p_0$")
        ax.plot(self.p_1_a, self.p_1_b, "r.", alpha=0.05, label="$p_1$")
        ax.set_xlabel("p-value for set 'a'")
        ax.set_ylabel("p-value for set 'b'")
        ax.legend()
        ax.set_aspect(1.0)
        return f


# %% [markdown]
# Now we compute the p-values with Mondrian Inductive

# %%
micp = MICP(sd)


# %%
# micp_panel = pn.Column(pn.Row(micp.sd.param, micp.sd.view),
#                       micp.view)
# %%

# %%
# micp_panel

# %%

# %%

# %%
def ECDF_comb(comb_func, ps, ps_cal):
    """Note: ps_cal are the p-values of the calibration examples with the same label as the p-value.
    Example: ECDF_comb(minimum, ps_test_0, ps_cal_0[y_cal==0])"""
    p_comb = comb_func(ps)
    ps_cal_comb = comb_func(ps_cal)
    return ECDF_cal_p(p_comb, ps_cal_comb)


# %%
def KolmogorovAveraging(p_vals, phi, phi_inv):
    return phi_inv(np.sum(phi(p_vals), axis=1) / p_vals.shape[1])


# %% [markdown]
# ## Arithmetic mean

# %%
def comb_arithmetic(ps, _=None):
    return np.mean(ps, axis=1)

def comb_arithmetic_ECDF(ps, ps_cal):
    return ECDF_comb(comb_arithmetic, ps, ps_cal)


# %%
# Unoptimized Irwin-Hall CDF
# Bates is the distribution of the mean of N independent uniform RVs
# Irwin-Hall is the distribution of the sum
from scipy.special import factorial, comb

def Irwin_Hall_CDF_base(x, n):
    acc = 0
    sign = 1
    for k in np.arange(0, np.floor(x) + 1):
        acc += sign * comb(n, k) * (x - k) ** n
        sign *= -1
    return acc / factorial(n)


Irwin_Hall_CDF = np.vectorize(Irwin_Hall_CDF_base, excluded=(1, "n"))

# %%
from functools import partial


def comb_arithmetic_q(ps, _=None):
    phi = lambda x: x.shape[1] * x
    phi_inv = partial(Irwin_Hall_CDF, n=ps.shape[1])
    return KolmogorovAveraging(ps, phi, phi_inv)


# %% [markdown]
# ## Geometric mean

# %%
import scipy.stats as ss

# %%
# %%
def comb_geometric(ps, _=None):
    return ss.gmean(ps, axis=1)

# %% [markdown]
# ## Fisher combination

# %%
def fisher(p, _=None):
    k = np.sum(np.log(p), axis=1).reshape(-1, 1)
    fs = -k / np.arange(1, p.shape[1]).reshape(1, -1)
    return np.sum(np.exp(
        k + np.cumsum(np.c_[np.zeros(shape=(p.shape[0])), np.log(fs)], axis=1)),
        axis=1)


# %%

def comb_geometric_ECDF(ps, ps_cal):
    return ECDF_comb(comb_geometric, ps, ps_cal)
# %%


# %% [markdown]
# ## Max p

# %%
def comb_maximum(ps, _=None):
    return np.max(ps, axis=1)


# %%
def comb_maximum_ECDF(ps, ps_cal):
    return ECDF_comb(comb_minimum, ps, ps_cal)
# %%
def comb_maximum_q(ps, _=None):
    max_ps = comb_maximum(ps)
    phi_inv = ss.beta(a=ps.shape[1], b=1).cdf

    return phi_inv(max_ps)

# %% [markdown]
# ## Minimum and Bonferroni

# %%
def comb_minimum(ps, _=None):
    return np.min(ps, axis=1)


# %%
def comb_minimum_ECDF(ps, ps_cal):
    return ECDF_comb(comb_minimum, ps, ps_cal)


# %% [markdown]
# The k-order statistic of n uniformly distributed variates is distributed as Beta(k,n+1-k).

# %%
def comb_minimum_q(ps, _=None):
    min_ps = comb_minimum(ps)
    phi_inv = ss.beta(a=1, b=ps.shape[1]).cdf

    return phi_inv(min_ps)

# %%
def comb_bonferroni(ps, _=None):
    return np.clip(ps.shape[1] * np.min(ps, axis=1), 0, 1)


# %%
def comb_bonferroni_q(ps, _=None):
    b_ps= p.shape[1] * np.min(ps, axis=1)
    phi_inv = ss.beta(a=1, b=ps.shape[1]).cdf

    return np.where(b_ps < 1.0 / ps.shape[1],
                    phi_inv(b_ps / ps.shape[1]),
                    1.0)


# %%
methodFunc = {"Arithmetic Mean": comb_arithmetic,
              "Arithmetic Mean (quantile)": comb_arithmetic_q,
              "Arithmetic Mean (ECDF)": comb_arithmetic_ECDF,
              "Geometric Mean": comb_geometric,
              "Geometric Mean (quantile)": fisher, # comb_geometric_q,
              "Geometric Mean (ECDF)": comb_geometric_ECDF,
              "Minimum": comb_minimum,
              "Bonferroni": comb_bonferroni,
              "Minimum (quantile)": comb_minimum_q,
              "Minimum (ECDF)": comb_minimum_ECDF,
              }


# %%

# %%
def cp_cm_widget(p_0, p_1, y):
    c_cm = cpConfusionMatrix_df(p_0, p_1, y).groupby('epsilon').agg('mean')
    c_cm['Actual error rate'] = c_cm[["Positive predicted Negative","Negative predicted Positive",
                                      "Positive predicted Empty","Negative predicted Empty"]].sum(axis=1) / c_cm.sum(axis=1)
    c_cm['Avg set size'] = (c_cm[["Positive predicted Negative", "Negative predicted Positive", 
                                  "Positive predicted Positive", "Negative predicted Negative"]].sum(axis=1) + \
                           2*(c_cm[["Positive predicted Uncertain","Negative predicted Uncertain"]].sum(axis=1))) / c_cm.sum(axis=1)
    cw = 50
    col_widths = {'epsilon': 50,
                  'Actual error rate': cw,
                  'Avg set size': cw,
                  "Positive predicted Positive": cw,
                  "Positive predicted Negative": cw,
                  "Negative predicted Negative": cw,
                  "Negative predicted Positive": cw,
                  "Positive predicted Empty": cw,
                  "Negative predicted Empty": cw,
                  "Positive predicted Uncertain": cw,
                  "Negative predicted Uncertain": cw}
#    return pn.widgets.DataFrame(c_cm, fit_columns=False, widths=col_widths,
#                                disabled=True)
    return pn.Pane(c_cm.to_html(notebook=True))


# %%
class SimpleCombination(param.Parameterized):
    sd = param.Parameter(precedence=-1)
    micp = param.Parameter(precedence=-1)
    p_comb_0 = param.Array(precedence=-1)
    p_comb_1 = param.Array(precedence=-1)

    method = param.Selector(list(methodFunc.keys()))

    def __init__(self, sd, micp, **params):
        self.sd = sd
        self.micp = micp
        super(SimpleCombination, self).__init__(**params)
        self.update()

    @pn.depends("micp.p_0_a", "micp.p_1_a", "micp.p_0_b", "micp.p_1_b",
                "method", watch=True)
    def update(self):
        comb_method = methodFunc[self.method]
        y_pcal = self.sd.output['y_pcal']
        ps_0 = np.c_[self.micp.p_0_a, self.micp.p_0_b]
        ps_pcal_0 = np.c_[self.micp.p_0_a_cal[y_pcal == 0], self.micp.p_0_b_cal[y_pcal == 0]]
        ps_1 = np.c_[self.micp.p_1_a, self.micp.p_1_b]
        ps_pcal_1 = np.c_[self.micp.p_1_a_cal[y_pcal == 1], self.micp.p_1_b_cal[y_pcal == 1]]

        with param.batch_watch(self):
            self.p_comb_0 = comb_method(ps_0, ps_pcal_0)
            self.p_comb_1 = comb_method(ps_1, ps_pcal_1)
    
    @pn.depends("p_comb_0", "p_comb_1")
    def view_table(self):
        return cp_cm_widget(self.p_comb_0, self.p_comb_1,
                            self.sd.output['y_test'])

    @pn.depends("p_comb_0", "p_comb_1")
    def view_validity(self):
        f = Figure()
        ax = f.add_subplot(1, 1, 1)
        ax.plot(*ecdf(self.p_comb_0[self.sd.output['y_test'] == 0]))
        ax.plot(*ecdf(self.p_comb_1[self.sd.output['y_test'] == 1]))
        ax.plot((0, 1), (0, 1), "k--")
        ax.set_aspect(1.0)
        ax.set_xlabel("Target error rate")
        ax.set_ylabel("Actual error rate")

        return f

        

# %%
sc = SimpleCombination(sd, micp)


# %%

# %%
class App(param.Parameterized):
    sd = param.Parameter()
    micp = param.Parameter()
    sc = param.Parameter()

    def __init__(self, sd, micp, fisher, **params):
        self.sd = sd
        self.micp = micp
        self.sc = sc

    @pn.depends("sc.p_comb_0", "sc.p_comb_1", watch=True)
    def view(self):
        return pn.Column(pn.Row(sd.param, sd.view),
                         pn.Row(micp.view_tables, micp.view_p_plane),
                         pn.Row(sc.param, sc.view_table, sc.view_validity))


# %%

# %%
if 0:
    app = App(sd, micp, sc)
    app.view()


# %%

# %%
# ss.pearsonr(p_0_a,p_0_b),ss.pearsonr(p_1_a,p_1_b)

# %%

# %%
class MultiCombination(param.Parameterized):
    sd = param.Parameter(precedence=-1)
    micp = param.Parameter(precedence=-1)
    p_comb_0 = param.Array(precedence=-1)
    p_comb_1 = param.Array(precedence=-1)

    methods_names = ["Base A", "Base B",] +  list(methodFunc.keys())
    methods = param.ListSelector(default=[methods_names[0]],objects=methods_names)

    def __init__(self, sd, micp, **params):
        self.sd = sd
        self.micp = micp
        super().__init__(**params)
        self.update()

    @pn.depends("micp.p_0_a", "micp.p_1_a", "micp.p_0_b", "micp.p_1_b",
                "methods", watch=True)
    def update(self):
        k = len(self.methods)
        p_comb_0 = np.zeros(shape=(k, micp.p_0_a.shape[0]))
        p_comb_1 = np.zeros(shape=(k, micp.p_0_a.shape[0]))
        ps_0 = np.c_[self.micp.p_0_a, self.micp.p_0_b]
        ps_1 = np.c_[self.micp.p_1_a, self.micp.p_1_b]
        y_pcal = self.sd.output['y_pcal']
        for i,m in enumerate(self.methods):
            if m=="Base A":
                p_comb_0[i] = self.micp.p_0_a
                p_comb_1[i] = self.micp.p_1_a
                continue
            elif m=="Base B":
                p_comb_0[i] = self.micp.p_0_b
                p_comb_1[i] = self.micp.p_1_b
                continue
            # If not a base CP, do the combination
            try:
                comb_method = methodFunc[m]
            except TypeError:
                comb_method = methodFunc[m[0]]
            ps_pcal_0 = np.c_[self.micp.p_0_a_cal[y_pcal == 0], self.micp.p_0_b_cal[y_pcal == 0]]
            ps_pcal_1 = np.c_[self.micp.p_1_a_cal[y_pcal == 1], self.micp.p_1_b_cal[y_pcal == 1]]
            p_comb_0[i] = comb_method(ps_0, ps_pcal_0)
            p_comb_1[i] = comb_method(ps_1, ps_pcal_1)
        with param.batch_watch(self):
            self.p_comb_0 = p_comb_0
            self.p_comb_1 = p_comb_1
    
    @pn.depends("p_comb_0", "p_comb_1")
    def view_table(self):
        return cp_cm_widget(self.p_comb_0, self.p_comb_1,
                            self.sd.output['y_test'])

    @pn.depends("p_comb_0", "p_comb_1")
    def view_validity(self):
        f = Figure(figsize=(12,12))
        ax = f.add_subplot(2, 2, 1)
        for i,m in enumerate(self.methods):
            ax.plot(*ecdf(self.p_comb_0[i][self.sd.output['y_test'] == 0]), label=m)
        ax.set_aspect(1.0)
        ax.set_xlabel("Target error rate")
        ax.set_ylabel("Actual error rate")
        ax.set_title("Validity plot for combined $p_0$")
        ax.legend()
        ax.plot((0, 1), (0, 1), "k--")

        ax = f.add_subplot(2, 2, 2)
        for i,m in enumerate(self.methods):
            ax.plot(*ecdf(self.p_comb_1[i][self.sd.output['y_test'] == 1]), label=m)
        ax.set_aspect(1.0)
        ax.set_xlabel("Target error rate")
        ax.set_ylabel("Actual error rate")
        ax.set_title("Validity plot for combined $p_1$")
        ax.legend()
        ax.plot((0, 1), (0, 1), "k--")
        
        ax = f.add_subplot(2, 2, 3)
        for i,m in enumerate(self.methods):
            ps = np.r_[self.p_comb_0[i],self.p_comb_1[i]]
            x,c = ecdf(ps)
                       
            ax.plot(x,2*(1-c), label=m)
        ax.set_xlabel("Target error rate")
        ax.set_xlim(0,1)
        ax.set_ylim(0,2)
        ax.set_ylabel("Average set size")
        ax.set_title("Combined CP efficiency")
        ax.legend()
        ax.plot((0, 1), (1, 0), "k--")
        ax.grid()

        ax = f.add_subplot(2, 2, 4)
        for i,m in enumerate(self.methods):
            ps = np.r_[self.p_comb_0[i],self.p_comb_1[i]]
            x,c = ecdf(ps)
                       
            ax.plot(x,2*(1-c)-(1-x), label=m)
        ax.set_xlabel("Target error rate")
        ax.set_xlim(0,1)
        ax.set_ylim(-1,1)
        ax.set_ylabel("Delta from ideal")
        ax.set_title("Combined CP efficiency (delta from ideal)")

        ax.grid()
        return f
    
mc = MultiCombination(sd, micp)


# %%
class AppMulti(param.Parameterized):
    sd = param.Parameter()
    micp = param.Parameter()
    mc = param.Parameter()

    def __init__(self, sd, micp, mc, **params):
        self.sd = sd
        self.micp = micp
        self.mc = mc

    def view(self):
        
        custom_mc_widgets = pn.Param(self.mc.param, widgets={"methods": pn.widgets.CheckBoxGroup})
        return pn.Column(pn.Row(self.sd.param, self.sd.view),
                         # pn.Row(self.micp.view_tables, self.micp.view_p_plane),
                         pn.Row(custom_mc_widgets, self.mc.view_validity))



# %%
am = AppMulti(sd,micp,mc)

# %%
ui = am.view()

# %%
srv = ui.show()

# %%
srv.stop()


# %% [markdown]
# # Neyman-Pearson

# %%
def BetaKDE(X, b):  # Unfortunately this is too slow in this implementation
    def kde(x):
        return sum(
            ss.beta(x / b + 1, (1 - x) / b + 1).pdf(x_i) for x_i in X) / len(X)

    return kde


# %% [markdown]
# ## Density estimation via histogram

# %%
def NeymanPearson(p_a, p_b, h0, test_p_a, test_p_b, pics_title_part):
    n_bins = 1000
    min_h1_lh = 0.0001

    f = plt.figure(figsize=(22, 5))

    ax = f.add_subplot(2, 4, 1)
    h, bins, _ = ax.hist([p_a[h0], p_a[~h0]],
                         bins=np.linspace(0, 1, n_bins + 1))
    ax.set_title("Histogram of p-values (a)")

    safe_h1 = np.where(h[1] == 0, min_h1_lh, h[1])
    lmbd_a = h[0] / safe_h1
    ax = f.add_subplot(2, 4, 2)
    ax.plot(bins[:-1], lmbd_a)
    ax.set_title("Lambda (a)")

    lmbd_a_interp = UnivariateSpline(
        np.concatenate(([0], 0.5 * (bins[1] - bins[0]) + bins[:-1])),
        np.concatenate(([0], lmbd_a)), k=1, s=0, ext=3)
    ax = f.add_subplot(2, 4, 3)
    ax.plot(np.linspace(0, 0.5, 101), lmbd_a_interp(np.linspace(0, 0.5, 101)))
    ax.set_title("Lambda (a) for p-values in [0,0.5]")

    ax = f.add_subplot(2, 4, 5)
    h, bins, _ = ax.hist([p_b[h0], p_b[~h0]],
                         bins=np.linspace(0, 1, n_bins + 1))
    ax.set_title("Histogram of p-values (b)")

    safe_h1 = np.where(h[1] == 0, min_h1_lh, h[1])
    lmbd_b = h[0] / safe_h1

    ax = f.add_subplot(2, 4, 6)
    ax.plot(bins[:-1], lmbd_b)
    ax.set_title("Lambda (b)")

    lmbd_b_interp = UnivariateSpline(
        np.concatenate(([0], 0.5 * (bins[1] - bins[0]) + bins[:-1])),
        # Let's add the origin and let's assume the middle of the bin
        np.concatenate(([0], lmbd_b)), k=1, s=0, ext=3)
    ax = f.add_subplot(2, 4, 7)
    ax.plot(np.linspace(0, 0.5, 101), lmbd_b_interp(np.linspace(0, 0.5, 101)));
    ax.set_title("Lambda (b) for p-values in [0,0.5]")

    lmbd_comb = lmbd_a_interp(p_a) * lmbd_b_interp(p_b)

    v, q = ecdf(lmbd_comb[h0])

    NP_calibr = UnivariateSpline(v, q, k=1, s=0, ext=3)

    lmbd_comb_test = lmbd_a_interp(test_p_a) * lmbd_b_interp(test_p_b)

    p_npcomb = NP_calibr(lmbd_comb_test)

    ax = f.add_subplot(1, 4, 4)
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    zz = NP_calibr(
        lmbd_a_interp(xx.ravel()) * lmbd_b_interp(yy.ravel())).reshape(xx.shape)

    ax.contourf(xx, yy, zz);
    ax.set_title("Combination of p-values")
    ax.set_xlabel("p (a)")
    ax.set_ylabel("p (b)")

    f.tight_layout()

    f.savefig(pics_base_name + pics_title_part + "_npcomb.png", dpi=300)

    return p_npcomb


# %%
p_0_npcomb = NeymanPearson(p_0_a_cal, p_0_b_cal, y_cal == 0, p_0_a, p_0_b,
                           pics_title_part="_0")


# %% [markdown]
# ## Density estimation via histogram smoothed with a spline

# %%
def splineEst(data, n_knots=20, s=0.3):
    k = np.linspace(0, 1, n_knots + 1)

    # UnivariateSpline() below requires that the x be strictly increasing
    # quantiles might be the same...

    h, bins = np.histogram(data, bins=n_knots, density=True)

    ss = UnivariateSpline(0.5 * (bins[:-1] + bins[1:]), h, k=3, s=s, ext=3)
    return ss


# %%
def NeymanPearsonDE(p_a, p_b, h0, p_a_test, p_b_test, pics_title_part,
                    densityEstimator=splineEst):
    f = plt.figure(figsize=(22, 5))

    p_a_h0 = p_a[h0]

    kde = densityEstimator(p_a_h0)
    l_h0 = kde(p_a)

    p_a_h1 = p_a[~h0]

    kde = densityEstimator(p_a_h1)
    l_h1 = kde(p_a)

    lmbd_a = l_h0 / l_h1

    # lmbd_a = np.clip(lmbd_a,1e-10,1e+10)
    ax = f.add_subplot(2, 3, 1)

    ax.plot(p_a, l_h0, "r.", label="Null")
    ax.plot(p_a, l_h1, "b.", label="Alternate")
    ax.set_title('Likelihoods (a)')

    ax = f.add_subplot(2, 3, 2)
    ax.plot(p_a, lmbd_a, "g.")
    ax.set_title('Lambda (a)')

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

    # lmbd_a = np.clip(lmbd_a,1e-10,1e+10)
    ax = f.add_subplot(2, 3, 4)

    ax.plot(p_b, l_h0, "r.", label="Null")
    ax.plot(p_b, l_h1, "b.", label="Alternate")
    ax.set_xlabel("p value")
    ax.set_title('Likelihoods (b)')

    ax = f.add_subplot(2, 3, 5)

    ax.plot(p_b, lmbd_b, "g.")
    ax.set_title('Lambda (b)')
    ax.set_xlabel("p value")

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

    ax = f.add_subplot(1, 3, 3)
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    zz = NP_calibr(lmbd_a_int(xx.ravel()) * lmbd_b_int(yy.ravel())).reshape(
        xx.shape)

    ax.contourf(xx, yy, zz);
    ax.set_title("Combination of p-values")
    ax.set_xlabel("p (a)")
    ax.set_ylabel("p (b)")
    ax.set_aspect(1)

    f.tight_layout()

    f.savefig(pics_base_name + pics_title_part + "_npde.png", dpi=300)

    return p_npcomb


# %%
p_0_npde = NeymanPearsonDE(p_0_a_cal, p_0_b_cal, y_cal == 0, p_0_a, p_0_b,
                           pics_title_part="_0")


# %%
def plot_diag_pVals(p_vals, descs, h0, pics_title_part):
    n_bins = 200
    bins = np.linspace(0, 1, n_bins + 1)
    f, axs = plt.subplots(len(p_vals), 1, figsize=(15, 5 * len(p_vals)))

    for ax, p, d in zip(axs, p_vals, descs):
        ax.hist([p[h0], p[~h0]], bins=bins, density=True)
        ax.set_xlabel("$p_{%s}$" % d, fontsize=14)

    f.suptitle("Histograms of p" + pics_title_part + " values", fontsize=18,
               y=1.02)
    f.tight_layout()

    f.savefig(pics_title_part + "_hists.png", dpi=150);


# %%
p_1_npcomb = NeymanPearson(p_1_a_cal, p_1_b_cal, y_cal == 1, p_1_a, p_1_b,
                           pics_title_part='_1')

# %%

# %%
p_1_npde = NeymanPearsonDE(p_1_a_cal, p_1_b_cal, y_cal == 1, p_1_a, p_1_b,
                           pics_title_part='_1')

# %%
c_cf_npde, precision_npde = cp_statistics(p_0_npde, p_1_npde, None, None,
                                          y_test, "_npde",
                                          " Neyman-Pearson (Spline) Combination");

# %%
c_cf_npcomb, precision_npcomb = cp_statistics(p_0_npcomb, p_1_npcomb, None,
                                              None, y_test, "_npcomb",
                                              " Neyman-Pearson (Hist) Combination");

# %%

# %% [markdown]
# # V-Matrix Density Ratio Approach

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
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])', nopython=True,
     parallel=True, nogil=True)
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
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])', parallel=True,
     nogil=True)
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
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])', parallel=True,
     nogil=True)
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
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])', parallel=True,
     nogil=True)
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
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])', parallel=True,
     nogil=True)
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
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])', parallel=True,
     nogil=True)
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
@jit('float64[:,:](float64[:,:],float64[:,:],float64[:,:])', nopython=True,
     parallel=True, nogil=True)
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
                          diag=True):
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

# %%
p_0_a.shape

# %%
# %%time
np_kwargs = dict(g=1e-5, krnl=rbf_krnl(4.5),
                 v_matrix=v_mat_star_sigma_rev_approx)
p_0_npcomb_vm = NeymanPearson_VMatrix(p_0_a_cal, p_0_b_cal, y_cal == 0, p_0_a,
                                      p_0_b, **np_kwargs)
p_1_npcomb_vm = NeymanPearson_VMatrix(p_1_a_cal, p_1_b_cal, y_cal == 1, p_1_a,
                                      p_1_b, **np_kwargs)
c_cf_npcomb_vm, precision_f_npcomb_vm = cp_statistics(p_0_npcomb_vm,
                                                      p_1_npcomb_vm, None, None,
                                                      y_test, "_np_v",
                                                      " NP (V-Matrix)");

# %%

# %% [markdown]
# + NNP Ideal                                                           | NA  	 606	14	 549	20	0	0	1880	1931
# + g=1e-6,krnl=rbf_krnl(6),v_matrix=v_mat_star_sigma_rev_approx        | 0.01	 584	26	  67	21	0	0	1890	2412
# + g=1e-5,krnl=rbf_krnl(6),v_matrix=v_mat_star_sigma_rev_approx        | 0.01	 562	17	 335	18	0	0	1921	2147  ## But better above 0.01
# + g=1e-5,krnl=rbf_krnl(5),v_matrix=v_mat_star_sigma_rev_approx        | 0.01	 543	15	 549	18	0	0	1942	1933
# + g=1e-5,krnl=rbf_krnl(4.5),v_matrix=v_mat_star_sigma_rev_approx      | 0.01	 545	16	 554	19	0	0	1939	1927

# %%

# %%

# %%
ps_0 = np.c_[p_0_a, p_0_b]
ps_1 = np.c_[p_1_a, p_1_b]

ps_0_cal = np.c_[p_0_a_cal, p_0_b_cal]
ps_1_cal = np.c_[p_1_a_cal, p_1_b_cal]

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Base A

# %%
c_cf_a, precision_a = cp_statistics(p_0_a, p_1_a, None, None, y_test, "_a",
                                    " base CP a");

# %% [markdown]
# ## Base B

# %%
c_cf_b, precision_b = cp_statistics(p_0_b, p_1_b, None, None, y_test, "_a",
                                    " base CP a");


# %%

# %%
def confusion_matrices(c_cf, epsilons=(0.01, 0.05, 0.10)):
    idx = pd.IndexSlice

    for eps in epsilons:
        c_cf_eps = c_cf.loc[idx[:, eps], idx[:]]
        c_cf_eps.index = c_cf_eps.index.droplevel(1)
        c_cf_eps.index.name = "$\epsilon=%0.2f$" % eps

        name_part = ("_%0.2f" % eps).replace('.', '_')
        with open(pics_base_name + '_cf' + name_part + '.txt', "w") as mf:
            print(c_cf_eps.to_latex(), file=mf)

        display(c_cf_eps)

    # %%


p_plane_plot(p_0_a, p_1_a, y_test, "Conformal Predictor A", "_a")

# %%
p_plane_plot(p_0_npde, p_1_npde, y_test, "NNP combination", "_nnp")

# %%
p_plane_plot(p_0_f, p_1_f, y_test, "Fischer combination", "_f")

# %%
p_plane_plot(p_0_npidcomb, p_1_npidcomb, y_test, "NNP (ideal)", "_npid")

# %%
p_plane_plot(p_0_npcomb_vm, p_1_npcomb_vm, y_test, "NP (V-Matrix)", "_npvmat")

# %%
cfs_to_compare = [
    (c_cf_a, "a"),
    (c_cf_b, "b"),

    (c_cf_npid, "NNP Ideal"),
    (c_cf_npcomb, "NNP"),
    (c_cf_npde, "NNP (spline)"),
    (c_cf_npcomb_vm, "NP V-Matrix"),

    (c_cf_avg, "Arith"),
    (c_cf_geom, "Geom"),
    (c_cf_max, "Max"),
    (c_cf_min, "Min"),
    (c_cf_bonf, "Bonferroni"),

    (c_cf_avg_q, "Arithmetic (Quantile)"),
    (c_cf_f, "Geometric (Quantile) Fisher"),
    (c_cf_max_q, "Max (Quantile)"),
    (c_cf_min_q, "Min (Quantile)"),
    (c_cf_bonf_q, "Bonferroni (Quantile)"),

    (c_cf_avg_ECDF, "Arithmetic (ECDF)"),
    (c_cf_geom_ECDF, "Geometric (ECDF)"),
    (c_cf_f_ECDF, "Fisher (ECDF)"),
    (c_cf_max_ECDF, "Max (ECDF)"),
]

cfs, method_names = zip(*cfs_to_compare)

c_cf = pd.concat(cfs,
                 keys=method_names, names=("p-values", "epsilon"))
confusion_matrices(c_cf)

# %%
