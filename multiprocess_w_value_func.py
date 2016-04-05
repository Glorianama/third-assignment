import numpy as np
import multiprocessing
import scipy as sp
import scipy.stats
import parameters as param
from scipy.integrate import dblquad
from scipy.optimize import fminbound
from numpy import log
from scipy import interp

SAFE = param.SAFE_R
BETA = param.BETA
# DISTRIBUTION PARAM (AGGREGATE)
AG_MEAN = param.AG_MEAN
AG_STDEV = param.AG_STDE
MAX_VAL_AG = param.AG_MAXVAL
# DISTRIBUTION PARAM (idios. shocks)
ID_MEAN = param.IS_MEAN
ID_STDEV = param.IS_STDE
MAX_VAL_E = param.IS_MAXVAL
MIN_VAL_E = param.IS_MINVAL
# Number of iterations
N = param.N2
COST = param.COST_INT

wealth_axis = np.linspace(1e-6,param.K_MAX,np.ceil(param.K_MAX*0.75))
v_func = 76.9230769* log(wealth_axis) + 3569.764136

def aggPDF(x):
    return scipy.stats.norm(loc=AG_MEAN,scale=AG_STDEV).pdf(x)
def idiPDF(x):
    return scipy.stats.norm(loc=ID_MEAN,scale=ID_STDEV).pdf(x)

def w_integral(s,W_ax,Y_ax):
    function = lambda eps,theta,s: max(interp(s*(theta+eps),W_ax,Y_ax),
                                     interp(s*(theta+eps)-COST,W_ax,v_func))*aggPDF(theta)*idiPDF(eps)
    return dblquad(function,MIN_VAL_E,MAX_VAL_E,lambda x: 0, lambda x: MAX_VAL_AG, args=(s,))

def w_bellman_objective(values, outputArray,l,w_a,w):
    i = values[0]
    k = values[1]

    objective = lambda s,k: - log(k-s) - BETA * w_integral(s,wealth_axis,w)[0]
    s_star = fminbound(objective, 1e-12, k-1e-12, args=(k,))
    outputArray[i] = -objective(s_star,k)

def w_bellman_op(w):
    Tw = np.empty(wealth_axis.size)
    Tw_e = multiprocessing.Array('f', Tw)
    wealth_obj = [[item[0],item[1]] for item in enumerate(wealth_axis)]
    l = multiprocessing.Lock()
    workers = [multiprocessing.Process(target=w_bellman_objective, args=(element,Tw_e,
        l,wealth_axis,w)) for element in enumerate(wealth_obj)]
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    return Tw_e

w = 10*log(wealth_axis) + 10
for i in range(N):
    w = w_bellman_op(w)
np.savetxt('w_value_func.csv',w,delimiter=',')