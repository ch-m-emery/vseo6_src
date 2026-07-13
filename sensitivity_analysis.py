"""
sensitivity_analysis.py module : run a temporal sobol-based sensitivity analysis over a hydrological model
"""

import math
import numpy as np
from matplotlib import pyplot as plt

from SALib.analyze import sobol
from SALib.sample import saltelli

from .constants import QIN_TS, N_SIZE_NETWORK, N_ITER
from .river_model import RiverModel

class SensitivityAnalysis:
    """A class to run a Sobol-based sensitivity analysis experiment
    """

    def __init__(self):
        """Class constructor
        """

        self._nb_parameters = 0
        self._problem = {}
        self._base_sample_size = 2**10

        self._tot_sample_size = 0
        self._param_values = None

    def generate_sample(self):
        """
        """
        self._param_values = saltelli.sample(
            self._problem,
            self._base_sample_size
        )
        self._tot_sample_size = self._param_values.shape[0]


class SensitivityAnalysisRiver(SensitivityAnalysis):
    """A class to run a Sobol-based sensitivity analysis experiment over the river model
    """

    def __init__(self, bool_qin=False):
        """Class constructor
        """

        super().__init__()

        self._nb_parameters = 3
        self._bool_qin = False
        self._problem = {
            "num_vars": self._nb_parameters,
            "names": ["k", "x", "inflow"],
            "bounds": [[0.5, 2.5],
                       [0.1, 0.3],
                       [0.999, 1.001]]
        }
        if bool_qin:
            self._bool_qin = True
            self._problem = {
                "num_vars": self._nb_parameters,
                "names": ["k", "x", "inflow"],
                "bounds": [[0.5, 2.5],
                           [0.1, 0.3],
                           [0.75, 1.25]]
            }

        # Q sensitivity
        self._q_out_e = None
        self._si_q = None

        # H sensitivity
        self._h_out_e = None
        self._si_h = None

    def _run_model(self, k=None, mat_q_in_ts=QIN_TS):
        """Run the river model with the k-th sample of input parameters
        """

        my_model = RiverModel(par_k_in=self._param_values[k,0],
                              par_x_in=self._param_values[k,1])

        mat_q_in_ts_k = mat_q_in_ts.copy()
        mat_q_in_ts_k *= self._param_values[k,2]
        q_out_ts = my_model.run(mat_q_in_ts=mat_q_in_ts_k)
        h_out_ts = my_model.estimate_height(q_out_ts)

        return q_out_ts, h_out_ts

    def generate_outputs_ensemble(self, n_network=N_SIZE_NETWORK, n_iter=N_ITER):
        """Generate the full set of model output from the set of sampled input parameters
        :param nb_iter:
        :param p_in:
        :return:
        """

        self._q_out_e = np.zeros((n_network, n_iter + 1, self._tot_sample_size))
        self._h_out_e = np.zeros((n_network, n_iter + 1, self._tot_sample_size))

        for k in range(self._tot_sample_size):
            self._q_out_e[:, :, k], self._h_out_e[:, :, k] = self._run_model(k)

    def estimate_sobol_ensemble(self):
        """Estimate time-varying 1st order Sobol indices for simulated Q and simulated H
        """

        self._si_q = np.zeros((self._q_out_e.shape[0], self._q_out_e.shape[1], self._nb_parameters))
        self._si_h = np.zeros((self._h_out_e.shape[0], self._h_out_e.shape[1], self._nb_parameters))

        for i in range(self._q_out_e.shape[0]):
            for t in range(self._q_out_e.shape[1]):

                if np.var(self._q_out_e[i, t, :])>0.01:
                    try:
                        results = sobol.analyze(self._problem, self._q_out_e[i, t, :])
                        self._si_q[i,t,:] = results["S1"]
                    except ValueError:
                        pass

                try:
                    results = sobol.analyze(self._problem, self._h_out_e[i, t, :])
                    self._si_h[i,t,:] = results["S1"]
                except ValueError:
                    pass

    def plot_sensitivity_indices(self):
        """Plot sensitivity analysis results: model outputs vs sobol indices
        """

        l_fig = []
        l_axis = []
        for i in range(N_SIZE_NETWORK):

            fig, axis = plt.subplots(2, 2)
            fig.suptitle(f"Reach {i + 1} sensitivity")

            # Q - model ensemble
            for k in range(self._tot_sample_size):
                axis[0, 0].plot(self._q_out_e[i, :, k], linestyle="-", color=(0.5, 0.5, 0.5), linewidth=0.5)
            axis[0, 0].plot(np.mean(self._q_out_e[i, :, :], 1), '-k')
            axis[0, 0].set_title("Model runs")
            axis[0, 0].set_ylabel("Simulated discharge")

            # H - model ensemble
            for k in range(self._tot_sample_size):
                axis[0, 1].plot(self._h_out_e[i, :, k], linestyle="-", color=(0.5, 0.5, 0.5), linewidth=0.5)
            axis[0, 1].plot(np.mean(self._h_out_e[i, :, :], 1), '-k')
            axis[0, 1].set_title("Model runs")
            axis[0, 1].set_ylabel("Simulated height")

            # Q - Sobol indices
            axis[1, 0].plot(self._si_q[i, :, 0], "-b", label="k")
            axis[1, 0].plot(self._si_q[i, :, 1], "-r", label="x")
            if self._bool_qin:
                axis[1, 0].plot(self._si_q[i, :, 2], "-g", label="inflow")
            axis[1, 0].set_title("Sobol indices for Q")
            axis[1, 0].set_ylabel("Time iterations")
            axis[1, 0].legend()

            # H - Sobol indices
            axis[1, 1].plot(self._si_h[i, :, 0], "-b", label="k")
            axis[1, 1].plot(self._si_h[i, :, 1], "-r", label="x")
            if self._bool_qin:
                axis[1,1].plot(self._si_h[i, :, 2], "-g", label="inflow")
            axis[1, 1].set_title("Sobol indices for H")
            axis[1, 1].set_ylabel("Time iterations")
            axis[1, 1].legend()

            l_fig.append(fig)
            l_axis.append(axis)

        plt.draw()
        plt.show()


if __name__ == "__main__":
    """Run model
    """

    my_analysis = SensitivityAnalysisRiver(bool_qin=True)
    my_analysis.generate_sample()
    my_analysis.generate_outputs_ensemble()
    my_analysis.estimate_sobol_ensemble()
    my_analysis.plot_sensitivity_indices()





