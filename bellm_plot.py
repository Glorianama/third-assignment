import plotly.offline as ply
import plotly.graph_objs as go
from numpy import genfromtxt
ply.init_notebook_mode()

pl_wealth_axis = genfromtxt('wealth_grid.csv',delimiter=',') # The k values of the mapping
pl_v_value_f = genfromtxt('v_value_func.csv',delimiter=',') # The mapped V(k) value function
pl_v_policy_f = genfromtxt('v_policy_func.csv',delimiter=',') # The V* greedy policy function
pl_w_value_f = genfromtxt('w_value_func.csv',delimiter=',') # The mapped W(k) value function
pl_w_policy_f = genfromtxt('w_policy_func.csv',delimiter=',') # The W* greedy policy function

## Plotting the  value functions

pl_v_val_trace = go.Scatter(
    x = pl_wealth_axis,
    y = pl_v_value_f,
    mode = 'lines+markers',
    name = 'V value function (not est.)',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,
    )
)
pl_w_val_trace = go.Scatter(
    x = pl_wealth_axis,
    y = pl_w_value_f,
    mode = 'lines+markers',
    name = 'W value function (est.)',
    line = dict(
        color = ('rgb(255, 165, 0)'),
        width = 4,
    )
)

data_val_func = [pl_v_val_trace, pl_w_val_trace]
layout_value_f = dict(title = 'Value functions of the Bellman-equation',
              xaxis = dict(title = 'Value function',linewidth=1),
              yaxis = dict(title = 'Wealth',linewidth=1),
              )

fig_value_func = dict(data=data_val_func,layout=layout_value_f)	
ply.iplot(fig_value_func)

## Plotting the policy functions

pl_v_pol_trace = go.Scatter(
    x = pl_wealth_axis,
    y = pl_v_policy_f,
    mode = 'lines+markers',
    name = 'V policy function (est.)',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,
    )
)
pl_w_pol_trace = go.Scatter(
    x = pl_wealth_axis,
    y = pl_w_policy_f,
    mode = 'lines+markers',
    name = 'W policy function (est.)',
    line = dict(
        color = ('rgb(255, 165, 0)'),
        width = 4,
    )
)
data_policy_func = [pl_v_pol_trace,pl_w_pol_trace]
layout_policy_f = dict(title = 'Policy function determining the optimal saving decision',
              xaxis = dict(title = 'Saving',linewidth=1),
              yaxis = dict(title = 'Wealth',linewidth=1),
              )
fig_policy_func = dict(data=data_policy_func,layout=layout_policy_f)
ply.iplot(fig_policy_func)