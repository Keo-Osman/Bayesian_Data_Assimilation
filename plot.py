import plotly.graph_objects as go
import numpy as np 
def plot_N_variables(
        states: np.ndarray, observations: np.ndarray, true_states: np.ndarray,
        time_stamps: np.ndarray, num_variables: int, title, variable_names=None
    ):
    if variable_names == None:
        variable_names = [i for i in range(num_variables)]
    
    colours = ['red', 'blue', 'green', 'orange', 'purple']
    fig = go.Figure()   
    for i in range(0, num_variables):
        # True states
        fig.add_trace(go.Scatter(
            x=time_stamps, y=true_states[:, i],
            mode="lines", name=f'True {variable_names[i]}',
            line=dict(color=colours[i])
        ))
        # Predictions
        fig.add_trace(go.Scatter(
            x=time_stamps, y=states[:, i],
            mode="lines", name=f'Estimated {variable_names[i]}',
            line=dict(dash="dash", color=colours[i])
        ))
        # # Forecasts
        # fig.add_trace(go.Scatter(
        #     x=time_stamps, y=forecasts[:, i],
        #     mode="markers", name=f'Forcasted {variable_names[i]}',
        #     marker=dict(size=7, color=colours[i], symbol="cross"),
        #     visible='legendonly'
        # ))

        # Observations
        fig.add_trace(go.Scatter(
            x=time_stamps, y=observations[:, i],
            mode="markers", name=f'Observed {variable_names[i]}',
            marker=dict(size=6, color=colours[i]),
            visible='legendonly'
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="State Value",
        legend_title="Signal",
        template="plotly_white"
    )

    fig.show()
