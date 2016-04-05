# These are the parameters of the model

T = 100 	# number of periods
SAFE_R = 1.01 	# return on safe project 1%
IS_MEAN = 0  	# meean of the idiosyncratic shocks
IS_STDE = 0.04	# std. error of the distribution of i.s.
IS_MINVAL = -0.5	# Minimum value for the integration
IS_MAXVAL = 0.5	# Maximum value for the integration
WE_MEAN = 2		# mean of the initial wealth distr.
WE_STDE = 0.4	# std. dev. of the initial wealth distr.
AG_MEAN = 1.02	# aggregate shocks: mean of the distr.
AG_STDE = 0.0225	# aggr.shocks: std. dev. of the distr.
AG_MAXVAL = 2.5	# Maximum value for the integration
BETA = 0.987	# discount preference
COST_INT = 1.5	# one-time cost of joining the financial int.
SAMPLING = 0.1  # sampling rate within the fin. intermediary
N1 = 2	# Number of iterations - to estimate v(k) - 100
N2 = 2	# Number of iterations - to estimate w(k) - 125
K_MAX = 15  # Max value where V(K) and W(K) will be mapped. - see the value_func.


	# the number of iterations over the  Bellman-equations are in the 
	# corresponding modules