# These are the parameters of the model

T = 100 	# number of periods
SAFE_R = 1.01 	# return on safe project 1%
IS_MEAN = 0  	# meean of the idiosyncratic shocks
IS_STDE = 0.04	# std. error of the distribution of i.s.
WE_MEAN = 2		# mean of the initial wealth distr.
WE_STDE = 0.4	# std. dev. of the initial wealth distr.
AG_MEAN = 1 	# aggregate shocks: mean of the distr.
AG_STDE = 0.0225	# aggr.shocks: std. dev. of the distr.
BETA = 0.987	# discount preference
COST_INT = 1.5	# one-time cost of joining the financial int.
SAMPLING = 0.1  # sampling rate within the fin. intermediary

	# the number of iterations over the  Bellman-equations are in the 
	# corresponding modules