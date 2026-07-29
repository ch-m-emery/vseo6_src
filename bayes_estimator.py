"""
bayes_estimator.py: to perform a data assimilation experiment
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import minimize

from .constants import VEC_TOBS, VEC_T, T_OBS_STEP
from .constants import Q_T0, QIN_TS, K_PRIOR
from .river_model import RiverModel

class DAExperiment():
    """A class to perform Bayes-estimator-based data assimilation experiment
    """

    def __init__(self, forward_model=None, vec_t_model=VEC_T, vec_t_obs=VEC_TOBS, t_obs_step=T_OBS_STEP, dct_obs=None, dct_ctl=None):
        """Class constructor
        """

        self.model = forward_model
        self.dct_ctl = dct_ctl
        # {"experiment_type": "", "variable": ""}
        self.vec_t_model = vec_t_model

        self.dct_obs = dct_obs
        # {"yobs": [], "variable": "", "reach": []}
        self.vec_t_obs = vec_t_obs
        self.t_obs_step = t_obs_step

        self._sig_o = None
        self._sig_b = None

    @property
    def sig_o(self):
        """Observation error standard deviation
        """
        return self._sig_o

    @sig_o.setter
    def sig_o(self, in_sig_o):
        """Observation error standard deviation setter
        """
        if in_sig_o<0.:
            raise ValueError("sig_o value must be positive")
        self._sig_o = in_sig_o

    @property
    def sig_b(self):
        """Control error standard deviation
        """
        return self._sig_b

    @sig_b.setter
    def sig_b(self, in_sig_b):
        """Control error standard deviation setter
        """
        if in_sig_b<0.:
            raise ValueError("sig_b value must be positive")
        self._sig_b = in_sig_b

    def plot_model_vs_obs(self, title="Model vs observations"):
        """Plot the model input + output & observation : adapted to the training session 5-reaches network
        """

        # Get model outputs
        q_out_ts = self.model.run()
        h_out_ts = self.model.estimate_height(q_out_ts)
        flt_max_q = np.amax(q_out_ts)*1.1
        flt_max_h = np.amax(h_out_ts)*1.1

        # Initiate figures
        fig, axis = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle(title)

        # Fill figure subplots
        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            ax.plot(q_out_ts[k, :], "-b", label="Q free run")
            ax.set_ylabel("discharge")
            ax.set_ylim((0., flt_max_q))

            ax_bis = ax.twinx()
            ax_bis.plot(h_out_ts[k, :], linestyle="-", color=(0.75, 0., 0.75), label="H free run")

            ax_bis.set_ylabel("height")
            ax_bis.set_ylim((0., flt_max_h))

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    if self.dct_obs["variable"] == "q":
                        ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', markeredgecolor=(0., 0.0, 1.), label="Q obs")
                    elif self.dct_obs["variable"] == "h":
                        ax_bis.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', markeredgecolor=(0.5, 0.0, 0.5), label="H obs")

                    else:
                        raise ValueError("Unknown observation variable, must be 'h' or 'q'")

            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax_bis.get_legend_handles_labels()

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

    def perform_state_estimation_propagation(self, assim_iter=0, qe=QIN_TS, q_t0_in=Q_T0):
        """Perform the propagation step of a single cycle of state estimation
        """

        q_t0 = q_t0_in
        q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
        q_run = self.model.run(n_iter=self.t_obs_step,
                                vec_q_0=q_t0,
                                mat_q_in_ts=q_in)

        flt_max_q = max(np.amax(q_run[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step+1]),
                        np.amax(self.dct_obs["yobs"]))*1.1

        fig, axis = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle("State estimation - propagation step")

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            for t in VEC_TOBS:
                ax.plot(t*np.ones((2,)), np.array([0., flt_max_q]), "--k", linewidth=0.75)
            ax.plot(q_run[k, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step+1], "-b", label="prior run")
            ax.set_ylabel("discharge")
            ax.set_ylim((0., flt_max_q))
            ax.set_xlim((0., np.amax(self.vec_t_obs)))

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="Q obs")

            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis[i, j]
                    ax.set_visible(False)
        plt.draw()
        plt.show()

        return q_run[:,-1], q_run

    def perform_state_estimation_analysis(self, sig_b_in=None, sig_o_in=None, assim_iter=0, xb=None, q_run=None):
        """Perform the analysis step of a single cycle of state estimation
        """

        self.sig_b = sig_b_in
        self.sig_o = sig_o_in

        # Bayes estimator parameters
        B = self.sig_b ** 2 * np.eye(self.model.n_dim)
        R = self.sig_o ** 2

        H = np.zeros((1, self.model.n_dim))
        for e in self.dct_obs["reach"]:
            H[0,e-1] = 1
        K = B @ np.transpose(H) / (H @ B @ np.transpose(H) + R)

        # Analysis
        d = self.dct_obs["yobs"][assim_iter] - H @ xb
        xa = xb + K @ d

        flt_max_q = max(
            np.amax(q_run[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1]),
            np.amax(self.dct_obs["yobs"])) * 1.1

        # Plot
        fig, axis = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle("State estimation - analysis step")

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            for t in VEC_TOBS:
                ax.plot(t*np.ones((2,)), np.array([0., flt_max_q]), "--k", linewidth=0.75)
            ax.plot(q_run[k, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1], "-b",
                    label="prior run")
            ax.plot(VEC_TOBS[assim_iter], xa[k], "x-r", label="analysis")
            ax.set_ylabel("discharge")
            ax.set_ylim((0., flt_max_q))
            ax.set_xlim((0., np.amax(self.vec_t_obs)))

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="Q obs")

            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis[i, j]
                    ax.set_visible(False)
        plt.draw()
        plt.show()

        return xa

    def perform_state_estimation_cycling(self, assim_iter=0, xa=None, qe=QIN_TS, q_t0_in=Q_T0):
        """Perform the cycling step of a single cycle of state estimation
        """

        q_t0 = q_t0_in
        q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
        qb_run = self.model.run(n_iter=self.t_obs_step,
                               vec_q_0=q_t0,
                               mat_q_in_ts=q_in)

        assim_next = assim_iter + 1
        q_t0 = xa
        q_in = qe[:, assim_next * self.t_obs_step: assim_next * self.t_obs_step + self.t_obs_step]
        qa_run = self.model.run(n_iter=self.t_obs_step,
                                vec_q_0=q_t0,
                                mat_q_in_ts=q_in)

        flt_max_q = max(
            np.amax(qb_run[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1]),
            np.amax(qa_run[:, assim_next * self.t_obs_step: assim_next * self.t_obs_step + self.t_obs_step + 1]),
            np.amax(self.dct_obs["yobs"])) * 1.1

        fig, axis = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle("State estimation - propagation step")

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            for t in VEC_TOBS:
                ax.plot(t*np.ones((2,)), np.array([0., flt_max_q]), "--k", linewidth=0.75)
            ax.plot(qb_run[k, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1], "-b",
                    label="prior run")
            vec_t_qa = np.arange(start=assim_next * self.t_obs_step, stop=assim_next * self.t_obs_step + 12)
            ax.plot( vec_t_qa,
                     qa_run[k, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step+ 12],
                     "-r", label="cycling")
            ax.set_ylabel("discharge")
            ax.set_ylim((0., flt_max_q))
            ax.set_xlim((0., np.amax(self.vec_t_obs)))

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="Q obs")

            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis[i, j]
                    ax.set_visible(False)
        plt.draw()
        plt.show()

    def perform_param_propagation(self):
        """
        """

        # Get model outputs
        q_out_ts = self.model.run()
        h_out_ts = self.model.estimate_height(q_out_ts)
        flt_max_q = np.amax(q_out_ts) * 1.1
        flt_max_h = np.amax(h_out_ts) * 1.1

        # Initiate figures
        fig, axis = plt.subplots(3, 3, figsize=(15, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle("Parameter estimation, propagation")

        # Fill figure subplots
        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            if self.dct_obs["variable"] == "q":
                ax.plot(q_out_ts[k, :], "-b", label="Q free run")
                ax.set_ylabel("discharge")
                ax.set_ylim((0., flt_max_q))
            elif self.dct_obs["variable"] == "h":
                ax.plot(h_out_ts[k, :], "-b", label="H free run")
                ax.set_ylabel("height")
                ax.set_ylim((0., flt_max_h))
            else:
                raise ValueError

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="obs")


            handles1, labels1 = ax.get_legend_handles_labels()
            handles = handles1
            labels = labels1
            ax.legend(handles, labels, loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis[i, j]
                    ax.set_visible(False)
        plt.draw()
        plt.show()

        return q_out_ts, h_out_ts

    def perform_param_innovation(self, qb_in=None, hb_in=None):
        """
        :param qb_in:
        :param hb_in:
        :return:
        """

        if self.dct_obs["variable"] == "q":
            d = self.dct_obs["yobs"] - qb_in[self.dct_obs["reach"][0]-1,self.vec_t_obs-1]
        elif self.dct_obs["variable"] == "h":
            d = self.dct_obs["yobs"] - hb_in[self.dct_obs["reach"][0]-1,self.vec_t_obs-1]
        else:
            raise ValueError

        return d

    def perform_param_analysis(self, nb_obs_in=None, sig_b_in=None, sig_o_in=None):
        """
        :param nb_obs_in:
        :param d:
        :return:
        """

        # Check inputs
        if nb_obs_in > len(self.dct_obs["yobs"]):
            raise ValueError("Input number of observations too large.")
        if nb_obs_in <= 0:
            raise ValueError("Invalid input for nb_obs_in, must be strictly positive.")

        self.sig_b = sig_b_in
        self.sig_o = sig_o_in

        # Optimization
        bool_h=False
        if self.dct_obs["variable"]=="h":
            bool_h=True
        res = minimize(fun=lambda k: self._cost_4dvar(k, nb_obs=nb_obs_in, bool_h=bool_h),
                       x0=np.array([K_PRIOR]),
                       method="L-BFGS-B",
                       bounds=[(0.5, 1.5)])
        return res.x[0]

    def _cost_4dvar(self, k_in=None, nb_obs=None, bool_h=False):
        """
        :return:
        """

        J = 0.5*((k_in-self.model.par_k[0])/self.sig_b)**2

        tmp_model = RiverModel(par_k_in=k_in[0])
        q = tmp_model.run()
        state = q.copy()
        if bool_h:
            state = tmp_model.estimate_height(state)

        for il_obs in range(nb_obs):
            J += 0.5*((self.dct_obs["yobs"][il_obs] - state[self.dct_obs["reach"][0]-1,self.vec_t_obs[il_obs]-1])/self.sig_o)**2

        return J

    def perform_param_rerun(self, ka=None, qb=None, hb=None):
        """
        :param ka:
        :param bool_h:
        :return:
        """

        bool_h = False
        if self.dct_obs["variable"] == "h":
            bool_h = True

        analysis_model = RiverModel(par_k_in=ka)
        qa = analysis_model.run()
        if bool_h:
            ha = analysis_model.estimate_height(qa)

        flt_max = max(np.amax(qb),np.amax(qa),np.amax(self.dct_obs["yobs"])) * 1.1
        if bool_h:
            flt_max = max(np.amax(hb),np.amax(ha),np.amax(self.dct_obs["yobs"])) * 1.1

        fig, axis = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle("Parameter estimation - rerun (final) step")

        zb = qb.copy()
        za = qa.copy()
        str_ylabel = "discharge"
        if bool_h:
            zb = hb.copy()
            za = ha.copy()
            str_ylabel = "height"

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            for t in VEC_TOBS:
                ax.plot(t * np.ones((2,)), np.array([0., flt_max]), "--k", linewidth=0.75)
            ax.plot(zb[k,:], "-b",label="background run")

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="obs")

            ax.plot(za[k, :], "-r", label="analysis run")
            ax.legend(loc='lower right', fontsize=8)

            ax.grid(True, which='both', linestyle='--', alpha=0.6)
            ax.set_ylabel(str_ylabel)
            ax.set_ylim((0., flt_max))
            ax.set_xlim((0., np.amax(self.vec_t_obs)))

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis[i, j]
                    ax.set_visible(False)
        plt.draw()
        plt.show()

    def _perform_state_assim(self, qe=QIN_TS, q_t0_in=Q_T0, b_in=None):
        """Perform the full state estimation experiment
        """

        # Free run
        q_bck_out_ts = self.model.run()
        # Analysis run
        q_ana_out_ts = np.zeros_like(q_bck_out_ts)

        # Bayes estimator parameters
        B = self.sig_b ** 2 * np.eye(self.model.n_dim)
        if b_in is not None:
            B = b_in
        R = self.sig_o ** 2

        H = np.zeros((1, self.model.n_dim))
        for e in self.dct_obs["reach"]:
            H[0,e-1] = 1
        K = B @ np.transpose(H) / (H @ B @ np.transpose(H) + R)

        # Assimilation
        q_t0 = q_t0_in
        for assim_iter in range(self.vec_t_obs.size):

            # Propagation
            q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
            q_run = self.model.run(n_iter=self.t_obs_step,
                                   vec_q_0=q_t0,
                                   mat_q_in_ts=q_in)
            q_ana_out_ts[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step+1] = q_run

            # Analysis
            xb = q_run[:,-1]
            d = self.dct_obs["yobs"][assim_iter] - H @ xb
            xa = xb + K @ d

            # Cycling
            q_t0 = xa

        return q_bck_out_ts, q_ana_out_ts

    def perform_assim(self, sig_b_in=None, sig_o_in=None, b_in=None, bool_diag_obs=False):
        """
        """

        self.sig_b = sig_b_in
        self.sig_o = sig_o_in

        if self.dct_ctl["experiment_type"] == "state":
            q_bck_out_ts, q_ana_out_ts = self._perform_state_assim(b_in=b_in)
            self.plot_assim_state(q_bck_out_ts, q_ana_out_ts)
        elif self.dct_ctl["experiment_type"] == "parameter":
            raise NotImplementedError
        else:
            raise ValueError("Unknown experiment type")

    def plot_assim_state(self, q_bck_out_ts, q_ana_out_ts, title="Assimilation results - Discharge"):
        """Plot outputs of the full state estimation experiment
        """

        flt_max_q = max(np.amax(q_bck_out_ts)*1.1, np.amax(q_ana_out_ts)*1.1)

        fig, axis = plt.subplots(3, 3, figsize=(16, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle("State estimation - Full experiment")

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            ax.plot(q_bck_out_ts[k, :], "-b", label="free run")
            ax.plot(q_ana_out_ts[k, :], "-r", label="analysis run")
            ax.set_ylabel("discharge")
            flt_max_q = max(np.amax(q_bck_out_ts[k, :]) * 1.1, np.amax(q_ana_out_ts[k, :]) * 1.1)
            flt_min_q = min(np.amin(q_bck_out_ts[k, :]) * 0.9, np.amin(q_ana_out_ts[k, :]) * 0.9)
            ax.set_ylim((flt_min_q, flt_max_q))
            ax.set_xlim((0., np.amax(self.vec_t_obs)))

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="obs")

            ax.legend(loc='lower right', fontsize=8)
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

    free_run = RiverModel()
    dct_obs_1 = {
        "yobs": [28.031, 31.354, 28.918, 26.057, 25.365, 26.536, 29.751, 30.046, 26.028, 26.167],
        "variable": "q",
        "reach": [5]
    }
    dct_obs_2 = {
        "yobs": [2.345, 2.594, 2.340, 2.028, 1.921, 2.086, 2.391, 2.312, 2.029, 1.949],
        "variable": "h",
        "reach": [5]
    }
    dct_obs_3 = {
        "yobs": [21.910, 26.992, 24.554, 19.885, 17.863, 19.427, 26.036, 23.850, 20.655, 19.492],
        "variable": "q",
        "reach": [3]
    }

    # Full DA experiment - state
    dct_ctl = {"experiment_type": "parameter", "variable": "k"}
    dct_obs=dct_obs_2
    nb_obs=5
    sig_b = 0.25
    sig_o = 0.1

    my_assim = DAExperiment(forward_model=free_run,
                            dct_obs=dct_obs,
                            dct_ctl=dct_ctl)
    qb_out_ts, hb_out_ts = my_assim.perform_param_propagation()
    d = my_assim.perform_param_innovation(qb_in=qb_out_ts, hb_in=hb_out_ts)
    ka = my_assim.perform_param_analysis(nb_obs_in=nb_obs, sig_b_in=sig_b, sig_o_in=sig_o)
    my_assim.perform_param_rerun(ka=ka, qb=qb_out_ts, hb=hb_out_ts)

    # my_assim.sig_o = 0.5
    # my_assim.sig_b = 0.25
    # my_assim.perform_assim()
    # B_in = np.array([
    #     [SIG_B_Q**2., 0., 0., 0., SIG_B_Q**2.*0.0625],
    #     [0., SIG_B_Q**2., 0., 0., SIG_B_Q**2.*0.125],
    #     [0., 0., SIG_B_Q**2., 0., SIG_B_Q**2.*0.25],
    #     [0., 0., 0., SIG_B_Q**2., SIG_B_Q**2.*0.5],
    #     [SIG_B_Q**2.*0.0625, SIG_B_Q**2.*0.125, SIG_B_Q**2.*0.25, SIG_B_Q**2.*0.5, SIG_B_Q**2.]
    # ])
    # my_assim.perform_assim(b_in=B_in)
