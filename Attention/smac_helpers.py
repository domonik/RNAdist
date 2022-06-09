import ConfigSpace
from smac.runhistory.runhistory import RunHistory
from smac.utils.io.traj_logging import TrajLogger
import numpy as np
import plotly.graph_objs as go
from smac.utils.constants import MAXINT


import os

def load_folder(directory):
    runhist = os.path.join(directory, "runhistory.json")
    cs_file = os.path.join(directory, "configspace.json")
    traj_file = os.path.join(directory, "traj_aclib2.json")
    cs_data = open(cs_file).read()
    cs = ConfigSpace.read_and_write.json.read(cs_data)
    rh = RunHistory()
    rh.load_json(runhist, cs)
    trajectory = TrajLogger.read_traj_aclib_format(fn=traj_file, cs=cs)
    return cs, rh, trajectory



def is_pareto_efficient_simple(costs):
    """
    Plot the Pareto Front in our 2d example.

    source from: https://stackoverflow.com/a/40239615
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)

            # And keep self
            is_efficient[i] = True
    return is_efficient


def plot_pareto_from_runhistory(observations):
    """
    This is only an example function for 2d plotting, when both objectives
    are to be minimized
    """

    # find the pareto front
    efficient_mask = is_pareto_efficient_simple(observations)
    front = observations[efficient_mask]
    # observations = observations[np.invert(efficient_mask)]

    obs1, obs2 = observations[:, 0], observations[:, 1]
    front = front[front[:, 0].argsort()]

    # add the bounds
    x_upper = np.max(obs1)
    y_upper = np.max(obs2)
    front = np.vstack([[front[0][0], y_upper], front, [x_upper, np.min(front[:, 1])]])

    x_front, y_front = front[:, 0], front[:, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_front, y=y_front, line=dict(dash="dot", width=4)))
    fig.add_trace(go.Scatter(x=obs1, y=obs2, mode="markers"))
    fig.show()


def plot_pareto(directory):
    cs, rh, traj = load_folder(directory)
    cost = np.vstack([v[0] for v in rh.data.values()])
    to_del = np.where(cost[:, 0] == MAXINT)[0]
    cost = np.delete(cost, to_del, axis=0)
    print(cost)
    plot_pareto_from_runhistory(cost)

def print_incumbent_from_dir(directory):
    cs, rh, traj = load_folder(directory)
    incumbent = traj[-1]["incumbent"]
    print(incumbent)
    print(rh.get_cost(incumbent))

if __name__ == '__main__':
    dir = "SMAC_OUTPUT/run_1608637542/"
    #plot_pareto(dir)
    print_incumbent_from_dir(dir)

