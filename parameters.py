# These are the parameters of the model

T = 30	# number of periods
SAFE_R = 1.01 	# return on safe project 1%
IS_MEAN = 0  	# meean of the idiosyncratic shocks
IS_STDE = 0.04	# std. error of the distribution of i.s.
IS_MINVAL = -0.2	# Minimum value for the integration
IS_MAXVAL = 0.2	# Maximum value for the integration
WE_MEAN = 2		# mean of the initial wealth distr.
WE_STDE = 0.4	# std. dev. of the initial wealth distr.
AG_MEAN = 1.03	# aggregate shocks: mean of the distr.
AG_STDE = 0.0225	# aggr.shocks: std. dev. of the distr.
AG_MAXVAL = 1.15	# Maximum value for the integration.
AG_MINVAL =0.90	# Minimum value for the integration.
BETA = 0.987	# discount preference
COST_INT = 1.75	# one-time cost of joining the financial int.
SAMPLING = 0.1  # sampling rate within the fin. intermediary

# Number of iterations on the Bellman equations.
N1 = 70	# Number of iterations - to estimate v(k) - 70
N2 = 70	# Number of iterations - to estimate w(k) - 125
K_MAX = 15  # Max value where V(K) and W(K) will be mapped. - see the value_func.
