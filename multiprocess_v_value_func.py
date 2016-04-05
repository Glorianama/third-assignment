import numpy as np
import scipy as sp
import scipy.stats
import parameters as param
from scipy.integrate import quad
from scipy.optimize import fminbound
from numpy import log
from scipy import interp


SAFE = param.SAFE_R
BETA = param.BETA
# DISTRIBUTION PARAM OF THETA (Aggregate shock)
MEAN = param.AG_MEAN
STDEV = param.AG_STDE
MAX_VAL = param.AG_MAXVAL   # Max value of theta for the integration
# Number of iterations
N = param.N1

wealth_axis = np.linspace(1e-6,15,10)

def PDF(x):
    return scipy.stats.norm(loc=MEAN,scale=STDEV).pdf(x)

def v_integral(s,W_ax,Y_ax):
    return quad((lambda theta,sI : interp(sI*max(SAFE,theta),
                                          W_ax,Y_ax)*PDF(theta)),0,MAX_VAL,args=(s,),limit=100)
def v_bellman_objective(values, outputArray, l,w_a,v):
    i = values[0]
    k = values[1]
    
    objective = lambda s,k: - np.log(k-s) - BETA * v_integral(s,
                                                              w_a,
                                                              v)[0]
    s_star = fminbound(objective, 1e-12, k-1e-12, args=(k,))
    outputArray[i] = -objective(s_star,k)
    

def v_bellman_op(v):
    Tv = np.empty(wealth_axis.size)
    wealth_obj = [[item[0],item[1]] for item in enumerate(wealth_axis)]
    Tv_e = multiprocessing.Array('f', Tv)
    l = multiprocessing.Lock()
    workers = [multiprocessing.Process(target=v_bellman_objective, args=(element, Tv_e,
        l,wealth_axis,v)) for element in wealth_obj]
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    return Tv_e

if __name__ == '__main__':
    v = 5 * log(wealth_axis) - 25
    for i in range(N):
        print " >>>> Iteration No. ", i
        v = v_bellman_op(v)
    np.savetxt('v_value_func.csv',v, delimiter=",")
    np.savetxt('wealth_grid.csv',wealth_axis, delimiter=",")