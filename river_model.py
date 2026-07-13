"""
river_model.py module : Muskingum-Cunge river routing model over a river network
"""

import numpy as np
import copy
from matplotlib import pyplot as plt

from .constants import N_SIZE_NETWORK, MAT_NETWORK, N_LONGEST_PATH, K_PRIOR, X_PRIOR
from .constants import D_T, N_ITER, Q_T0, QIN_TS
from .constants import RC_ALPHA, RC_BETA

class RiverModel:
    """A class to simulate a river routing model based on the Muskingum-Cunge model
    """

    def __init__(self, n_dim=N_SIZE_NETWORK, mat_n_in=MAT_NETWORK, par_k_in=K_PRIOR, par_x_in=X_PRIOR, par_qin_mul=1.0):
        """Class constructor
        :param n_dim: int
        :param mat_n_in: 2D np.array
        :param par_k_in: int or 1D np.array
        :param par_x_in: int or 1D np.array
        :param par_qin_mul: float
        """

        # Set river network dimension
        if not isinstance(n_dim, int):
            raise TypeError("Input network size 'n_dim' must be an integer")
        if n_dim<=0:
            raise ValueError("Input network size 'n_dim' must be positive")
        self._n_dim = n_dim

        # Set river network connectivity
        if mat_n_in is None:
            raise ValueError("Missing input network matrix 'mat_n_in'")
        if mat_n_in.ndim != 2:
            raise ValueError("Input network matrix must be 2D")
        if mat_n_in.shape[0] != mat_n_in.shape[1]:
            raise ValueError("Input network matrix must be square")
        if mat_n_in.shape[0] != n_dim:
            raise ValueError("Input network matrix size does not match input network size")
        if np.all(np.tril(mat_n_in, k=0) == 0):
            raise ValueError("Input network matrix must be strictly lower triangular")
        self._mat_n_in = mat_n_in

        # Set k-parameter value
        if par_k_in is None:
            raise ValueError("Missing input parameter vector 'par_k_in'")
        if not isinstance(par_k_in, (np.ndarray, float)):
            raise TypeError(f"Input k-parameter vector must be a np.ndarray or int, got {par_k_in.__class__}")
        if isinstance(par_k_in, np.ndarray):
            if par_k_in.ndim != 1:
                raise ValueError("Input k-parameter vector must be 1D")
            if par_k_in.shape[0] != n_dim:
                raise ValueError("Input k-parameter vector size does not match input network size")
            self._par_k_in = par_k_in
        if isinstance(par_k_in, float):
            if par_k_in <= 0 :
                raise ValueError("Input k-parameter must be positive")
            self._par_k_in = par_k_in * np.ones((self._n_dim,))
            
        # Set x-parameter value
        if par_x_in is None:
            raise ValueError("Missing input parameter vector 'par_x_in'")
        if not isinstance(par_x_in, (np.ndarray, float)):
            raise TypeError(f"Input x-parameter vector must be a np.ndarray or int, got {par_x_in.__class__}")
        if isinstance(par_x_in, np.ndarray):
            if par_x_in.ndim != 1:
                raise ValueError("Input x-parameter vector must be 1D")
            if par_x_in.shape[0] != n_dim:
                raise ValueError("Input x-parameter vector size does not match input network size")
            self._par_x_in = par_x_in
        if isinstance(par_x_in, float):
            if par_x_in <= 0 :
                raise ValueError("Input k-parameter must be positive")
            self._par_x_in = par_x_in * np.ones((self._n_dim,))

        if par_qin_mul < 0.:
            raise ValueError("Input qin_mul parameter must be positive")
        if not isinstance(par_qin_mul, float):
            raise TypeError("Input qin_mul parameter must be a float")
        self._par_qin_mul = par_qin_mul

        self.qe = None
        self.qt0 = None

        self._vec_c1 = None
        self._vec_c2 = None
        self._vec_c3 = None
        self._model_mat = None
        
    def copy(self):
        """Copy method
        """
        return copy.deepcopy(self)

    @property
    def n_dim(self):
        """Number of reaches in the river network
        :return n_dim: int
        """
        return self._n_dim

    @property
    def par_k(self):
        """Storage time constant
        :return par_k: float
        """
        return self._par_k_in

    def _set_c1(self, d_t=D_T):
        """Set C1 parameters from river network parameters
        """
        self._vec_c1 = (0.5*d_t - self._par_k_in*self._par_x_in) / (self._par_k_in*(1-self._par_x_in) + 0.5*d_t)

    def _set_c2(self, d_t=D_T):
        """Set C2 parameters from river network parameters
        """
        self._vec_c2 = (0.5 * d_t + self._par_k_in * self._par_x_in) / (self._par_k_in * (1 - self._par_x_in) + 0.5 * d_t)

    def _set_c3(self, d_t=D_T):
        """Set C3 parameters from river network parameters
        """
        self._vec_c3 = (self._par_k_in * (1 - self._par_x_in) - 0.5 * d_t) / (self._par_k_in * (1 - self._par_x_in) + 0.5 * d_t)

    def _prepare_run(self, in_longest_path=N_LONGEST_PATH):
        """Muskingum-Cunge model matrix (Not optimized, can not be used on a big network)
        :param in_longest_path: int
            add a routine to automatically compute it
        :return:
        """

        self._model_mat = np.eye(self._n_dim)
        mat_propag = np.diag(self._vec_c1) @ self._mat_n_in
        mat_tmp = mat_propag
        for k in range(1,in_longest_path):
            self._model_mat += mat_tmp
            mat_tmp @= mat_propag
        self._model_mat += mat_tmp

    def _single_iteration(self, vec_q_in_t=None, vec_q_out_t=None):
        """Run a single model iteration
        :param vec_q_in_t:
        :param vec_q_out_t:
        :return:
        """

        vec_q_out_next = self._vec_c1 * vec_q_in_t
        vec_q_out_next += self._vec_c3 * vec_q_out_t
        vec_q_out_next += np.diag(self._vec_c2) @ (self._mat_n_in @ vec_q_out_t + vec_q_in_t)

        vec_q_out_next = self._model_mat @ vec_q_out_next

        return vec_q_out_next

    def run(self, n_iter=N_ITER, in_longest_path=N_LONGEST_PATH, d_t=D_T, vec_q_0=Q_T0, mat_q_in_ts=QIN_TS):
        """
        :param n_iter:
        :param in_longest_path:
        :param d_t:
        :param vec_q_0:
        :param mat_q_in_ts:
        """

        # Check inputs
        if vec_q_0.ndim != 1:
            raise ValueError("Initial condition should be a 1D vector")
        if vec_q_0.size != self._n_dim:
            raise ValueError("Initial condition does not match network size")

        if mat_q_in_ts.ndim != 2:
            raise ValueError("Inflow boundary condition should be a 2D vector")
        if mat_q_in_ts.shape[0] != self._n_dim:
            raise ValueError("Inflow boundary condition does not match network size")
        if mat_q_in_ts.shape[1] != n_iter:
            raise ValueError("Inflow boundary condition timeseries does not match number of iteration")

        # Compute routing parameters
        self._set_c1(d_t)
        self._set_c2(d_t)
        self._set_c3(d_t)

        # Prepare run
        self._prepare_run(in_longest_path=in_longest_path)
        q_out_ts = np.zeros((self._n_dim, n_iter+1))
        q_out_ts[:,0] = vec_q_0

        for il_i in range(n_iter):
            q_out_ts[:,il_i+1] = self._single_iteration(vec_q_in_t=mat_q_in_ts[:,il_i] * self._par_qin_mul,
                                                        vec_q_out_t=q_out_ts[:,il_i])

        return q_out_ts

    @staticmethod
    def rating_curve(h_in):
        """A synthetic rating curve relating elevation h in a reach to the simulated discharge
        :param h_in:
        :return:
        """
        q_out = (h_in / RC_ALPHA) ** (1./RC_BETA)
        return q_out

    @staticmethod
    def estimate_height(q_out_ts):
        """The inverted rating curve to get h from simulation q
        :param q_out_ts:
        :return:
        """

        # Estimate h
        h_out_ts = RC_ALPHA * q_out_ts ** (RC_BETA)

        return h_out_ts

    @staticmethod
    def plot(q_out_ts, h_out_ts, mat_q_in_ts=QIN_TS, title="Model outputs"):
        """Plot the model input + output : adapted to the training session 5-reaches network
        :param q_out_ts:
        :param h_out_ts:
        :param mat_q_in_ts:
        :param title:
        :return:
        """

        flt_max_q = np.amax(q_out_ts)*1.1
        flt_max_h = np.amax(h_out_ts)*1.1

        fig, axis = plt.subplots(3, 3, figsize=(12,9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1,2), (2, 2)]
        fig.suptitle(title)

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k+1}")

            ax.plot(mat_q_in_ts[k,:], linestyle="--", color=(0.0, 0.0, 0.75), linewidth=0.75, label="Qin")
            ax.plot(q_out_ts[k, :], "-b", label="Qout")
            handles1, labels1 = ax.get_legend_handles_labels()
            ax.set_ylabel("discharge")
            ax.set_ylim((0., flt_max_q))

            ax_bis = ax.twinx()
            ax_bis.plot(h_out_ts[k, :], linestyle="-", color=(0.75, 0., 0.75), label="Hout")
            handles2, labels2 = ax_bis.get_legend_handles_labels()
            ax_bis.set_ylabel("height")
            ax_bis.set_ylim((0., flt_max_h))

            handles = handles1 + handles2
            labels = labels1 + labels2
            ax.legend(handles, labels, loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis[i, j]
                    ax.set_visible(False)

        plt.draw()
        plt.show()


if __name__ == "__main__":
    """Main run
    """

    # Run model
    my_model = RiverModel()
    q_out_ts = my_model.run()
    h_out_ts = my_model.estimate_height(q_out_ts)
    my_model.plot(q_out_ts, h_out_ts)








