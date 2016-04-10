# Based on the approach taken by Sargent and Stachurski:
# http://quant-econ.net/py/optgrowth.html.

import numpy as np
import multiprocessing
import scipy as sp
import scipy.stats
import parameters as param
from scipy.integrate import dblquad
from scipy.optimize import fminbound
from numpy import log
from scipy import interp

# SAFE RETURN AND DISCOUNT FACTOR
SAFE = param.SAFE_R
BETA = param.BETA
# DISTRIBUTION PARAM (AGGREGATE shock)
AG_MEAN = param.AG_MEAN
AG_STDEV = param.AG_STDE
MAX_VAL_AG = param.AG_MAXVAL
MIN_VAL_AG = param.AG_MINVAL    
# DISTRIBUTION PARAM (idios. shocks)
ID_MEAN = param.IS_MEAN
ID_STDEV = param.IS_STDE
MAX_VAL_E = param.IS_MAXVAL
MIN_VAL_E = param.IS_MINVAL
# Number of iterations
N = param.N2
COST = param.COST_INT # Cost of joining the fin. interm.

wealth_axis = np.linspace(1e-6,param.K_MAX,2*param.K_MAX) # np.ceil(param.K_MAX*0.75)
# We assume the following value function for v(k) based on the original paper
v_func = 76.9230769* log(wealth_axis) + 3569.764136
# or if the result from the estimated v(k) is used:
# v_func = numpy.genfromtxt(v_value_func.csv, delimiter=',')

def aggPDF(x):
    """
        Returns the density of a given value from the distribution of the 
        aggregate shock <theta>.
    """
    return scipy.stats.norm(loc=AG_MEAN,scale=AG_STDEV).pdf(x)
def idiPDF(x):
    """
        Returns the density of a given value from the distribution of the
        idiosynratic shock <eps = epsilon>.
    """
    return scipy.stats.norm(loc=ID_MEAN,scale=ID_STDEV).pdf(x)

def w_integral(s,W_ax,Y_ax):
    """
        The integral part of the Bellman-equation.
    """
    function = lambda theta,eps,s: max(interp(s*(theta+eps),W_ax,Y_ax), \
        interp(s*(theta+eps)-COST,W_ax,v_func))*aggPDF(theta)*idiPDF(eps)
    return dblquad(function,MIN_VAL_E,MAX_VAL_E,lambda x: MIN_VAL_AG, lambda x: MAX_VAL_AG, args=(s,))

def w_bellman_objective(values,outputArray,l,w_a,w):
    """
        The maximazation task for each worker process at a given wealth stock/bequest<k>.
        Maximize the value function by the saving rate <s> assuming the value function form
        from the previous iteration. It returns the new new value at <k> in the new
        function form.

        The functional form is represented by the value of the function at each designated
        <k> and linear interpolation is used to get the value of the function for other
        <k>-s.
    """
    i = values[0]
    k = values[1]
    # Here we define the function to be optimized as a function of s and k.
    objective = lambda s,k: -log(k-s)-BETA*w_integral(s,wealth_axis,w)[0]
    # The optimal saving for a given k and w_.
    s_star = fminbound(objective,0.3*k,k-1e-12,args=(k,))
    # The new value at k in the iterated value function.
    outputArray[i] = -objective(s_star,k)

def w_bellman_op(w):
    """
        This function returns the new value function form <Tw_e> at a given iteration stage.
        It relies on the multiprocessing package that spawns 2*param.K_MAX number of
        processes to work simultanouosly, thus increasing the process rate somewhat
        (2*param.K_max = the number of wealth values where the function will be mapped).
    """
    wealth_obj = [[item[0],item[1]] for item in enumerate(wealth_axis)]
    Tw_e = multiprocessing.Array('f',np.empty(wealth_axis.size))
    l = multiprocessing.Lock()
    workers=[multiprocessing.Process(target=w_bellman_objective,args=(element,Tw_e,
        l,wealth_axis,w)) for element in wealth_obj]
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    return Tw_e

def policy(w):
    """
        Assuming that the value function takes the form stored in w,
        the function returns the w* greedy policy function, which gives
        the optimal saving decision for each level of wealth/bequest <k>.
    """
    policy_f = np.empty(wealth_axis.size)
    for i,k in enumerate(wealth_axis):
        objective = lambda s,k: -log(k-s)-BETA*w_integral(s,wealth_axis,w)[0]
        # Returning the optimal saving decision for a given k.
        policy_f[i] = fminbound(objective,1e-12,k-1e-12,args=(k,))
    return policy_f


if __name__=='__main__':
    # We start with an initial naive assumption on the form of the value function w.
    # We assume log-linear form given the form of v(k).
    w = 60*log(wealth_axis) + 1500
    # To follow how the estimated functional form changes, we log every functional form.
    time_path = [[] for i in range(N)]
    # Iteration on the Bellman equation.
    for i in range(N):
        print ">>>> Iteration No. ", i
        w = w_bellman_op(w)
        time_path[i] = w
    # Obtain the w* greedy policy function.
    greedy_policy = policy(v)

    # Export the estimated function and the log on the estimeted functions.
    # We assume that the wealth grid is already exported in wealth_grid.csv, otherwise:
    #np.savetxt('wealth_grid.csv',wealth_axis, delimiter=",")
    np.savetxt('w_value_func.csv',w,delimiter=',')
    print "Return value function w in 'w_value_func.csv'"
    np.savetxt('w_policy_func.csv',greedy_policy,delimiter=',')