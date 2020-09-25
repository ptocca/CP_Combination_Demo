# -*- coding: utf-8 -*-
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

pn.extension('mathjax',comm='ipywidgets')

# %%
# %matplotlib inline
from matplotlib.figure import Figure

# %%
import param

# %%
from CP import *


# %%
def plot_score_hist(alpha_0_a, alpha_0_b, alpha_1_a, alpha_1_b):
    f = Figure(figsize=(12, 4))
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
    cc = param.Number(default=0.0, bounds=(-0.99, 0.99))
    var = param.Number(default=1.0, bounds=(0.5, 4.0))

    # Outputs
    output = param.Dict(default=dict(),
                        precedence=-1)  # To have all updates in one go

    n = 2

    def __init__(self, **params):
        self.micp_calibration_fraction = 0.5
        self.comb_calibration_fraction = 0.3        
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
        except np.linalg.LinAlgError:
            placeholder = np.array([0.0, 1.0])
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

    @pn.depends("N", "percentage_of_positives", "seed", "cc", "var")
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


# %%
# MICP

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


# %%
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
def ECDF_comb(comb_func, ps, ps_cal, h0_cal):
    p_comb = comb_func(ps)
    ps_cal_comb = comb_func(ps_cal[h0_cal])
    return ECDF_cal_p(p_comb, ps_cal_comb)


# %%
def KolmogorovAveraging(p_vals, phi, phi_inv):
    return phi_inv(np.sum(phi(p_vals), axis=1) / p_vals.shape[1])


# %%
## Arithmetic mean

# %%
def comb_arithmetic(ps,  *unused):
    return np.mean(ps, axis=1)

def comb_arithmetic_conservative(ps,  *unused):
    return np.clip(2*np.mean(ps, axis=1), a_min=0.0, a_max=1.0)

def comb_arithmetic_ECDF(ps,  ps_cal, h0_cal):
    return ECDF_comb(comb_arithmetic, ps, ps_cal, h0_cal)


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


def comb_arithmetic_q(ps,  *unused):
    phi = lambda x: x.shape[1] * x
    phi_inv = partial(Irwin_Hall_CDF, n=ps.shape[1])
    return KolmogorovAveraging(ps, phi, phi_inv)


# %%
## Geometric mean

# %%
import scipy.stats as ss

# %%
# %%
def comb_geometric(ps,  *unused):
    return ss.gmean(ps, axis=1)

# %%
def comb_geometric_conservative(ps,  *unused):
    return np.clip(np.e*ss.gmean(ps, axis=1), a_min=0.0, a_max=1.0)

# %%
## Fisher combination

# %%
def fisher(p, *unused):
    k = np.sum(np.log(p), axis=1).reshape(-1, 1)
    fs = -k / np.arange(1, p.shape[1]).reshape(1, -1)
    return np.sum(np.exp(
        k + np.cumsum(np.c_[np.zeros(shape=(p.shape[0])), np.log(fs)], axis=1)),
        axis=1)

# %%

def comb_geometric_ECDF(ps,  ps_cal, h0_cal):
    return ECDF_comb(comb_geometric, ps, ps_cal, h0_cal)
# %%
## Max p

# %%
def comb_maximum(ps,  *unused):
    return np.max(ps, axis=1)


# %%
def comb_maximum_ECDF(ps, ps_cal, h0_cal):
    return ECDF_comb(comb_minimum, ps, ps_cal, h0_cal)
# %%
def comb_maximum_q(ps,  *unused):
    max_ps = comb_maximum(ps)
    phi_inv = ss.beta(a=ps.shape[1], b=1).cdf

    return phi_inv(max_ps)

# %%
## Minimum and Bonferroni

# %%
def comb_minimum(ps,  *unused):
    return np.min(ps, axis=1)


# %%
def comb_minimum_ECDF(ps,  ps_cal, h0_cal):
    return ECDF_comb(comb_minimum, ps, ps_cal, h0_cal)


# %%
# The k-order statistic of n uniformly distributed variates is distributed as Beta(k,n+1-k).

# %%
def comb_minimum_q(ps,  *unused):
    min_ps = comb_minimum(ps)
    phi_inv = ss.beta(a=1, b=ps.shape[1]).cdf

    return phi_inv(min_ps)

# %%
def comb_bonferroni(ps,  *unused):
    return np.clip(ps.shape[1] * np.min(ps, axis=1), 0, 1)


# %%
def comb_bonferroni_q(ps,  *unused):
    b_ps= p.shape[1] * np.min(ps, axis=1)
    phi_inv = ss.beta(a=1, b=ps.shape[1]).cdf

    return np.where(b_ps < 1.0 / ps.shape[1],
                    phi_inv(b_ps / ps.shape[1]),
                    1.0)


# %%
import NP_Combine

import VMatrix_Combine

# %%
methodFunc = {"Arithmetic Mean": comb_arithmetic,
              "Arithmetic Mean (conservative)": comb_arithmetic_conservative,
              "Arithmetic Mean (quantile)": comb_arithmetic_q,
              "Arithmetic Mean (ECDF)": comb_arithmetic_ECDF,
              
              "Geometric Mean": comb_geometric,
              "Geometric Mean (conservative)": comb_geometric_conservative,
              "Geometric Mean (quantile)": fisher, # comb_geometric_q,
              "Geometric Mean (ECDF)": comb_geometric_ECDF,
              
              "Minimum": comb_minimum,
              "Bonferroni": comb_bonferroni,
              "Minimum (quantile)": comb_minimum_q,
              "Minimum (ECDF)": comb_minimum_ECDF,

              "Naive Neyman-Pearson": NP_Combine.comb_nnp,
              "V-Matrix Neyman-Pearson": VMatrix_Combine.comb_np_vm,
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


# %%
# ss.pearsonr(p_0_a,p_0_b),ss.pearsonr(p_1_a,p_1_b)

# %%

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.palettes import brewer
from bokeh.models import ColumnDataSource, HoverTool

# %%
def ColorCycler(i):
    return brewer['Dark2'][8][i%8]

# %%
class MultiCombination(param.Parameterized):
    sd = param.Parameter(precedence=-1)
    micp = param.Parameter(precedence=-1)
    p_comb_0 = param.Array(precedence=-1)
    p_comb_1 = param.Array(precedence=-1)

    methods_names = ["Base A", "Base B",] +  list(methodFunc.keys())
    methods = param.ListSelector(default=[methods_names[0], methods_names[1]],objects=methods_names)

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
            ps_pcal_0 = np.c_[self.micp.p_0_a_cal, self.micp.p_0_b_cal]
            ps_pcal_1 = np.c_[self.micp.p_1_a_cal, self.micp.p_1_b_cal]
            p_comb_0[i] = comb_method(ps_0, ps_pcal_0, y_pcal==0)
            p_comb_1[i] = comb_method(ps_1, ps_pcal_1, y_pcal==1)
        with param.batch_watch(self):
            self.p_comb_0 = p_comb_0
            self.p_comb_1 = p_comb_1
    
    @pn.depends("p_comb_0", "p_comb_1")
    def view_table(self):
        return cp_cm_widget(self.p_comb_0, self.p_comb_1,
                            self.sd.output['y_test'])

    @pn.depends("p_comb_0", "p_comb_1")
    def view_validity(self, p_w = 500, p_h = 500):
        ax_v_0 = figure(title="Validity plot for combined p₀", plot_width=p_w, plot_height=p_h)

        for i,m in enumerate(self.methods):
            x,y = ecdf(self.p_comb_0[i][self.sd.output['y_test'] == 0])
            ax_v_0.line(x=x, y=y, legend_label=m, color=ColorCycler(i), line_width=2)
        ax_v_0.legend.location = 'top_left'
        ax_v_0.xaxis.axis_label = "Target error rate"
        ax_v_0.yaxis.axis_label = "Actual error rate"
        ax_v_0.xaxis.bounds = (0,1)
        ax_v_0.yaxis.bounds = (0,1)

        ax_v_1 = figure(title="Validity plot for combined p₁", plot_width=p_w, plot_height=p_h)
        for i,m in enumerate(self.methods):
            x,y = ecdf(self.p_comb_1[i][self.sd.output['y_test'] == 1])
            ax_v_1.line(x, y, legend_label=m, color=ColorCycler(i), line_width=2)
        ax_v_1.legend.location = 'top_left'

        ax_v_1.xaxis.axis_label = "Target error rate"
        ax_v_1.yaxis.axis_label = "Actual error rate"
        ax_v_1.xaxis.bounds = (0,1)
        ax_v_1.yaxis.bounds = (0,1)

        ax_v_1.line(x=(0,1), y=(0,1), line_color="black", name='Ideal', line_dash="dashed")
        
        tooltips = [("Significance level","@x")]
        ax_eff = figure(title="Efficiency of combined CP", tools="box_zoom,reset,save", tooltips=tooltips, plot_width=p_w, plot_height=p_h)
        #ax_eff = figure(title="Efficiency of combined CP", plot_width=p_w, plot_height=p_h)
        
        cds_df = pd.DataFrame(columns=['x'])
        for i,m in enumerate(self.methods):
            ps = np.r_[self.p_comb_0[i],self.p_comb_1[i]]
            x,c = ecdf(ps)                                    ### Remind me why this is right
            # To compute the average set size as a function of the significance, we need to find
            # the fraction of p0 and p1 are greater than the significance
            # The ECDF is 1 minus this fraction.
            # We could compute separately the ECDF for p0 and p1 and then we sum them.
            # We can also compute the ECDF of the union of the p0 and p1
            cds_df = pd.merge_ordered(cds_df, pd.DataFrame({'x':x, m:2*(1-c)}), how='outer', fill_method='ffill', on='x')
        
            tooltips.append((m,"@{%s}"%m))
        
        for i,m in enumerate(self.methods):        
            ax_eff.line(x='x', y='%s'%m, source=cds_df, legend_label=m, color=ColorCycler(i), line_width=2)
            
        # ax_eff.add_tools(HoverTool(mode='vline', tooltips=tooltips, names=self.methods, line_policy='nearest'))
        
        
            
        ax_eff.xaxis.axis_label = "Target error rate"
        ax_eff.yaxis.axis_label = "Average prediction set size"
        ax_eff.xaxis.bounds = (0,1)
        ax_eff.yaxis.bounds = (0,2)
        ax_eff.line(x=(0,1), y=(1,0), line_color="black", name='Ideal', line_dash="dashed")
        

        ax_eff_d = figure(title="Delta from ideal efficiency", plot_width=p_w, plot_height=p_h)
        for i,m in enumerate(self.methods):
            ps = np.r_[self.p_comb_0[i],self.p_comb_1[i]]
            x,c = ecdf(ps)
                       
            ax_eff_d.line(x=x,y=2*(1-c)-(1-x), legend_label=m, color=ColorCycler(i), line_width=2)
        # ax.set_xlabel("Target error rate")
        # ax.set_xlim(0,1)
        # ax.set_ylim(-1,1)
        # ax.set_ylabel("Delta from ideal")
        # ax.set_title("Combined CP efficiency (delta from ideal)")

        # ax.grid()

        return gridplot([[ax_v_0, ax_v_1],
                         [ax_eff, ax_eff_d]])
    
mc = MultiCombination(sd, micp)


# %%
class AppMulti(param.Parameterized):

    sd = param.Parameter()
    micp = param.Parameter()
    mc = param.Parameter()

    def __init__(self, sd, micp, mc, **params):
        super().__init__(**params)
        self.sd = sd
        self.micp = micp
        self.mc = mc

        self.ui = self.create_ui()

        self.ui_elem_selector.param.watch(self.update_UI, "value")
        self.ui_elem_selector.param.trigger('value')

    def update_UI(self, event):
        ui = []
        if "Synthetic Dataset" in event.new:
            ui.append(self.sd_view)
        if "Base CPs" in event.new:
            ui.append(self.micp_view)
        if "Combination" in event.new:
            ui.append(self.comb_view)

        self.ui[1] = pn.Column(*ui)

    def create_ui(self):
        ui_components_names = ["Synthetic Dataset", "Base CPs", "Combination"]
        self.ui_elem_selector = pn.widgets.CheckBoxGroup(
            value=[ui_components_names[0], ui_components_names[2]], options=ui_components_names)
        custom_mc_widgets = pn.Param(self.mc.param, widgets={
                                     "methods": pn.widgets.CheckBoxGroup})
        self.sd_view = pn.Row(self.sd.param, self.sd.view)
        self.micp_view = pn.Row(self.micp.view_tables, self.micp.view_p_plane)
        self.comb_view = pn.Row(custom_mc_widgets, self.mc.view_validity)

        return pn.Row(self.ui_elem_selector, pn.Column(self.sd_view, self.micp_view, self.comb_view))

    def view(self):
        return self.ui

    



# %%
am = AppMulti(sd,micp,mc)

# %%
ui = am.view()

# %%
notes = pn.pane.LaTeX(r"""
<h1>Notes</h1>
This demo accompanies the paper <a href="http://proceedings.mlr.press/v105/toccaceli19a/toccaceli19a.pdf"> 'CP Combination via Neyman-Pearson Lemma'</a> presented at COPA2019 as well as Chapter 6 of P. Toccaceli PhD Thesis.<br>

It allows a user to experiment with various methods for the combination of Conformal Predictors (in the case of binary classification).


<h2>Structure of the demo</h2>
The demo considers two sets of NCMs, computed by two different hypothetical underlying ML methods denoted with A and B, on the same set of observations.<br>
The NCMs are generated as variates of two Gaussians.<br>
The NCMs for observations of class 'Negative' are centered on -1 and those of the class 'Positive' on +1.<br>
The demo has three components that can be displayed or hidden via the tickboxes at the left.<br>
The "Synthetic Dataset" panel allows the user to choose interactively the parameters governing the generation of the dataset of NCMs.
The user can choose the relative proportions of the two classes ('Negative' and 'Positive'), the initialization of the Pseudo-Random Number Generator, 
the correlation coefficient and the variance.<br>
The 'Base CPs' panels shows the CP confusion matrices for the base CPs arising from the NCM generated as synthetic dataset. 
This table is mainly for completeness.<br>
The 'Combination' panel presents a choice a combination methods and displays the validity and efficiency in four plots. 
The top row contains two validity plots, one for the 'Negative' class and one for the 'Positive' class. In the row below, the average set size 
(as function of the target error rate) is plotted. The diagram on the right presents the delta between the achieved average set size and the ideal one. <br>
<br>
NOTE: The V-Matrix Neyman-Pearson method can be slow, especially for data set sizes larger than 5000.


<h2>Brief reminder of the theory</h2>
We recall briefly the key points (please to the sources mentioned above for the details).<br>
 we consider fixed methods and adaptive methods. The fixed methods are summarised in the table below.
Generally, the combination functions result in loss of validity, whereas the merging function preserve it but at the cost of efficiency.
It is possible to recover validity if one knows the CDF of the combined CP. Indeed, for some combination functions, such CDF is known, under
the assumption of independence.

\[ %\renewcommand{\arraystretch}{2}
    \begin{array}{c|c|c|c}
    & \text{Combination function} & \text{Merging function} & \text{Combination function CDF} \\
    \text{Arithmetic average} &  p_{arith\_avg} = \frac{1}{d}\sum_{i=1}^{d}p^{(i)} &  2 \cdot p_{arith\_avg} & \frac{1}{n!} \sum_{k=0}^{ \left \lfloor {t} \right \rfloor} {(-1)}^k {d \choose k} {(t-k)}^{d} \\
    \text{Geometric average}  &  p_{geom\_avg} = {\left ( \prod_{i=1}^{d}p^{(i)} \right )}^{\frac{1}{d}}  &  e \cdot p_{geom\_avg}  & t\sum_{i=0}^{d-1}\frac{(-\log t)^i}{i!} \\
    \text{Min}               &  p_{min} = \min(p^{(1)},\dots,p^{(d)})   &  d \cdot p_{min}  & \text{Beta}(d,1) \\
    \text{Max }              &   p_{max} = \max(p^{(1)},\dots,p^{(d)})     & p_{max}  & \text{Beta}(1,d)
    \end{array} \]

It can be argued that the assumption of independence is not realistic.
The p-values come from CP with different underlying algorithms, but refer to the same object.<br>

The dependence among p-values can follow patterns that cannot be predicted a priori. So, adaptive methods are considered. 
One approach is to reserve part of the training set to create a calibration set for the combined p-values. The ECDF of the combined p-values of 
this calibration combination set is used to calibrate the combined p-values for the test set.<br>

In the demo, the merging functions are denoted with '(conservative)', the calibration via CDF with '(quantile)' and the 'adaptive' calibration with '(ECDF)'. <br>

The adaptive approach, while resulting in (practically) valid p-values at the cost of sacrificing part of the training set, still suffers from the possibility that the combination law might not be optimal 
(in the sense of achieving the best efficiency).

We use the Neyman-Pearson Lemma to combine the p-values so that the efficiency is optimal.

The most powerful test between two simple hypothesis $H_0: \theta = \theta_0$ and $H_1: \theta = \theta_1$ is the one that uses as test statistic the likelihood ratio:
\[ \Lambda(x) := \frac{\mathcal{L}(\theta_0 | x)}{\mathcal{L}(\theta_1 | x)} \]
and as threshold the value $\eta$ that satisfies
\[
\epsilon = \mathbb{P}\left[ \Lambda(X) \leq \eta \; \middle| \; H_0 \right]
\]
where $\epsilon$ is the significance level.

""")

ui_and_notes = pn.Tabs(("CP Combinations",ui),
              ("Notes",notes))
ui_and_notes                # render inline for use with 'appmode'
#%%
# srv = ui_and_notes.show()

# %%
# srv.stop()

# %%
# srv1 = pn.Row(mc.view_validity(p_w = 500, p_h=500)).show()
# %%
# srv1.stop()
