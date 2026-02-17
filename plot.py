import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 
def plot_N_variables(
        states: np.ndarray, observations: np.ndarray, true_states: np.ndarray,
        absolute_error: np.ndarray, percent_error: np.ndarray,
        time_stamps: np.ndarray, num_variables: int, title, variable_names=None
    ):
    if variable_names == None:
        variable_names = [i for i in range(num_variables)]
    
    colours = ['red', 'blue', 'green', 'orange', 'purple']
    fig = make_subplots(rows=3, cols=1)   
    for i in range(0, num_variables):
        # True states
        fig.add_trace(go.Scatter(
            x=time_stamps, y=true_states[:, i],
            mode="lines", name=f'True {variable_names[i]}',
            line=dict(color=colours[i])
        ), row=1, col=1)
        # Predictions
        fig.add_trace(go.Scatter(
            x=time_stamps, y=states[:, i],
            mode="lines", name=f'Estimated {variable_names[i]}',
            line=dict(dash="dash", color=colours[i])
        ), row=1, col=1)
        # Observations
        fig.add_trace(go.Scatter(
            x=time_stamps, y=observations[:, i],
            mode="markers", name=f'Observed {variable_names[i]}',
            marker=dict(size=6, color=colours[i]),
            visible='legendonly'
        ), row=1, col=1)
        # Absolute error
        fig.add_trace(go.Scatter(
            x=time_stamps, y=absolute_error[:, i],
            mode="lines", name=f'Absolute error {variable_names[i]}',
            line=dict(color=colours[i])
        ), row=2, col=1)
        # Percent error
        fig.add_trace(go.Scatter(
            x=time_stamps, y=percent_error[:, i],
            mode="lines", name=f'% error {variable_names[i]}',
            line=dict(color=colours[i])
        ), row=3, col=1)
        
    fig.update_layout(
        height = 2000,
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="State Value",
        legend_title="Signal",
        template="plotly_white"
    )

    fig.show()
