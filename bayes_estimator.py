"""
bayes_estimator.py: to perform a data assimilation experiment
"""

import numpy as np

from matplotlib import pyplot as plt

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

    def perform_state_estimation_analysis(self, assim_iter=0, xb=None, q_run=None):
        """Perform the analysis step of a single cycle of state estimation
        """

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

    def perform_parameter_estimation_propagation(self, assim_iter=0, qe=QIN_TS, q_t0_in=Q_T0, bool_diag_obs=False):
        """Perform the propagation step of a single cycle of parameter estimation
        """

        # Control vector
        xb = self.model.par_k[0]

        # Initiate figure for parameter evolution
        fig_par, axis_par = plt.subplots(1,1)
        fig_par.suptitle("[parameter] Parameter estimation - propagation step")

        # Plot all observations
        for t_obs in VEC_TOBS:
            axis_par.plot(t_obs*np.ones((2,)), np.array([0., 1.5]), '--k', linewidth=0.75)
        axis_par.plot(np.array([0., VEC_TOBS[assim_iter]]), xb*np.ones((2,)), "-b", label="prior")
        axis_par.set_ylabel("k parameter")
        axis_par.set_xlim((0., np.amax(self.vec_t_obs)))
        axis_par.set_ylim((0., 1.5))

        # Propagate model
        q_t0 = q_t0_in
        q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
        q_run = self.model.run(n_iter=self.t_obs_step,
                                vec_q_0=q_t0,
                                mat_q_in_ts=q_in)
        flt_max_q = max(
            np.amax(q_run[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1]),
            np.amax(self.dct_obs["yobs"])) * 1.1
        if bool_diag_obs:
            h_run = self.model.estimate_height(q_run)
            flt_max_h = max(
                np.amax(h_run[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1]),
                np.amax(self.dct_obs["yobs"])) * 1.1


        # Initiate figure for state evolution
        fig_model, axis_model = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig_model.suptitle("[model] Parameter estimation - propagation step")

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis_model[i, j]
            ax.set_title(f"Reach {k + 1}")

            if bool_diag_obs:
                for t_obs in VEC_TOBS:
                    ax.plot(t_obs * np.ones((2,)), np.array([0., flt_max_h]), '--k', linewidth=0.75)
                ax.plot(h_run[k, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step+1], "-b", label="prior run")
                ax.set_ylabel("height")
                ax.set_ylim((0., flt_max_h))
            else:
                for t_obs in VEC_TOBS:
                    ax.plot(t_obs * np.ones((2,)), np.array([0., flt_max_q]), '--k', linewidth=0.75)
                ax.plot(q_run[k, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step+1], "-b", label="prior run")
                ax.set_ylabel("discharge")
                ax.set_ylim((0., flt_max_q))

            ax.set_xlim((0., np.amax(self.vec_t_obs)))

            str_labl_obs = "Q obs"
            if bool_diag_obs:
                str_labl_obs = "H obs"
            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label=str_labl_obs)

            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis_model[i, j]
                    ax.set_visible(False)
        plt.draw()
        plt.show()

        out_run = q_run
        if bool_diag_obs:
            out_run = h_run

        return xb, out_run

    def perform_parameter_estimation_analysis(self, assim_iter=0, xb=None, q_run=None):
        """Perform the analysis step of a single cycle of parameter estimation
        """

        # Bayes estimator parameters
        B = self.sig_b ** 2
        R = self.sig_o ** 2

        H = np.zeros((1, self.model.n_dim))
        for e in self.dct_obs["reach"]:
            H[0, e - 1] = 1
        K = (B / (B + R))

        # Analysis
        d = self.dct_obs["yobs"][assim_iter] - H @ q_run[:, -1]
        xa = xb + K * d[0]
        if xa < 0:
            print("Warning: negative parameter value after assimilation")

        # Initiate figure for parameter evolution
        fig_par, axis_par = plt.subplots(1, 1)
        fig_par.suptitle("[parameter] Parameter estimation - analysis step")

        for t_obs in VEC_TOBS:
            axis_par.plot(t_obs * np.ones((2,)), np.array([0., 1.5]), '--k', linewidth=0.75)
        axis_par.plot(np.array([0., VEC_TOBS[assim_iter]]), xb * np.ones((2,)), "-b", label="prior")
        axis_par.plot(np.array([0., VEC_TOBS[assim_iter]]), xa * np.ones((2,)), "-r", label="analysis")
        axis_par.set_ylabel("k parameter")
        axis_par.set_xlim((0., np.amax(self.vec_t_obs)))
        axis_par.set_ylim((0., 1.5))

        plt.draw()
        plt.show()

        return xa

    def perform_parameter_estimation_cycling(self, assim_iter=0, xb=None, xa=None, q_run=None, qe=QIN_TS, q_t0_in=Q_T0, bool_diag_obs=False):
        """Perform the cycling step of a single cycle of parameter estimation
        """

        # Cycling
        q_t0 = q_t0_in
        q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
        model_assim = RiverModel(par_k_in=xa)
        qa_run = model_assim.run(n_iter=self.t_obs_step,
                                vec_q_0=q_t0,
                                mat_q_in_ts=q_in)
        flt_max_q = max(
            np.amax(q_run[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1]),
            np.amax(qa_run),
            np.amax(self.dct_obs["yobs"])) * 1.1

        if bool_diag_obs:
            ha_run = self.model.estimate_height(qa_run)
            flt_max_h = max(
                np.amax(ha_run[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1]),
                np.amax(q_run),
                np.amax(self.dct_obs["yobs"])) * 1.1

        # Initiate figure for parameter evolution
        fig_par, axis_par = plt.subplots(1, 1)
        fig_par.suptitle("[parameter] Parameter estimation - cycling step")

        for t_obs in VEC_TOBS:
            axis_par.plot(t_obs * np.ones((2,)), np.array([0., 1.5]), '--k', linewidth=0.75)
        axis_par.plot(np.array([0., VEC_TOBS[assim_iter]]), xb * np.ones((2,)), "-b", label="prior")
        axis_par.plot(np.array([0., VEC_TOBS[assim_iter]]), xa * np.ones((2,)), "-r", label="analysis")
        axis_par.set_ylabel("k parameter")
        axis_par.set_xlim((0., np.amax(self.vec_t_obs)))
        axis_par.set_ylim((0., 1.5))

        # Initiate figure for state evolution
        fig_model, axis_model = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig_model.suptitle("[model] Parameter estimation - cycling step")

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis_model[i, j]
            ax.set_title(f"Reach {k + 1}")

            ax.plot(q_run[k, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1], "-b",
                    label="prior run")
            if not bool_diag_obs:
                for t_obs in VEC_TOBS:
                    ax.plot(t_obs * np.ones((2,)), np.array([0., flt_max_q]), '--k', linewidth=0.75)
                ax.plot(qa_run[k, :], "-r",
                        label="analysis run")
                ax.set_ylabel("discharge")
                ax.set_ylim((0., flt_max_q))
            else:
                for t_obs in VEC_TOBS:
                    ax.plot(t_obs * np.ones((2,)), np.array([0., flt_max_h]), '--k', linewidth=0.75)
                ax.plot(ha_run[k, :], "-r",
                        label="analysis run")
                ax.set_ylabel("height")
                ax.set_ylim((0., flt_max_h))

            ax.set_xlim((0., np.amax(self.vec_t_obs)))

            str_labl_obs = "Q obs"
            if bool_diag_obs:
                str_labl_obs = "H obs"
            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label=str_labl_obs)

            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis_model[i, j]
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

    def _perform_param_assim(self, qe=QIN_TS, q_t0_in=Q_T0, bool_diag_obs=False):
        """Perform the full parameter estimation experiment
        """

        # Free run
        mod_bck_out_ts = self.model.run()
        if bool_diag_obs:
            mod_bck_out_ts = self.model.estimate_height(mod_bck_out_ts)
        # Analysis run
        mod_ana_out_ts = np.zeros_like(mod_bck_out_ts)

        # Bayes estimator parameters
        B = self.sig_b ** 2
        R = self.sig_o ** 2
        H = np.zeros((1, self.model.n_dim))
        for e in self.dct_obs["reach"]:
            H[0,e-1] = 1
        K = (B / (B + R))

        # Assimilation
        xb = np.zeros((VEC_TOBS.size,))
        xb[0] = K_PRIOR
        xa = np.zeros((VEC_TOBS.size,))

        # Assimilation
        q_t0 = q_t0_in
        model_assim = self.model.copy()
        for assim_iter in range(self.vec_t_obs.size):

            # Propagation
            q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
            mod_run = model_assim.run(n_iter=self.t_obs_step,
                                   vec_q_0=q_t0,
                                   mat_q_in_ts=q_in)
            if bool_diag_obs:
                mod_run = model_assim.estimate_height(mod_run)

            # Analysis
            d = self.dct_obs["yobs"][assim_iter] - H @ mod_run[:, -1]
            xa[assim_iter] = xb[assim_iter] + K * d[0]

            if xa[assim_iter] < 0:
                print("Warning: negative parameter value after assimilation")

            # Cycling
            model_assim = RiverModel(par_k_in=xa[assim_iter])
            q_run = model_assim.run(n_iter=self.t_obs_step,
                                   vec_q_0=q_t0,
                                   mat_q_in_ts=q_in)
            h_run = model_assim.estimate_height(q_run)
            mod_ana_out_ts[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1] = q_run
            if bool_diag_obs:
                mod_ana_out_ts[
                    :, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1] = h_run
            q_t0 = q_run[:, -1]
            try:
                xb[assim_iter+1] = xa[assim_iter]
            except IndexError:
                pass

        # Final analysis run
        model_analysis = RiverModel(par_k_in=xa[-1])
        mod_final = model_analysis.run()
        if bool_diag_obs:
            mod_final = model_analysis.estimate_height(mod_final)


        return xa, mod_bck_out_ts, mod_ana_out_ts, mod_final

    def perform_assim(self, b_in=None, bool_diag_obs=False):
        """
        """

        if self.dct_ctl["experiment_type"] == "state":
            q_bck_out_ts, q_ana_out_ts = self._perform_state_assim(b_in=b_in)
            self.plot_assim_state(q_bck_out_ts, q_ana_out_ts)
        elif self.dct_ctl["experiment_type"] == "parameter":
            xa, q_bck_out_ts, q_ana_out_ts, q_final = self._perform_param_assim(bool_diag_obs=bool_diag_obs)
            self.plot_assim_param(xa, q_bck_out_ts, q_ana_out_ts, q_final, bool_diag_obs=bool_diag_obs)
        else:
            raise ValueError("Unknown experiment type")

    def plot_assim_state(self, q_bck_out_ts, q_ana_out_ts, title="Assimilation results - Discharge"):
        """Plot outputs of the full state estimation experiment
        """

        flt_max_q = max(np.amax(q_bck_out_ts)*1.1, np.amax(q_ana_out_ts)*1.1)

        fig, axis = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle("State estimation - Full experiment")

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            ax.plot(q_bck_out_ts[k, :], "-b", label="free run")
            ax.plot(q_ana_out_ts[k, :], "-r", linewidth=0.75, label="analysis run")
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

    def plot_assim_param(self, xa, q_bck_out_ts, q_ana_out_ts, q_final, bool_diag_obs=False, title="Parameter estimation - Full experiment"):
        """Plot outputs of the full parameter estimation experiment
        """

        # Initiate figure for parameter evolution
        fig_par, axis_par = plt.subplots(1, 1)
        fig_par.suptitle("[parameter] Parameter estimation - analysis step")

        for t_obs in VEC_TOBS:
            axis_par.plot(t_obs * np.ones((2,)), np.array([0., 1.5]), '--k', linewidth=0.75)
        axis_par.plot(np.array([0., VEC_TOBS[-1]]), K_PRIOR * np.ones((2,)), "-b", label="prior")
        for il_t, t_obs in enumerate(VEC_TOBS[:-1]):
            axis_par.plot(np.array([t_obs-T_OBS_STEP, t_obs]), xa[il_t] * np.ones((2,)), linestyle="--", color=(1.0, 0.5, 0.), label="analysis step")
        axis_par.set_ylabel("k parameter")
        axis_par.set_xlim((0., np.amax(self.vec_t_obs)))
        axis_par.set_ylim((0., 1.5))

        flt_max_q = max(np.amax(q_bck_out_ts) * 1.1, np.amax(q_ana_out_ts) * 1.1)

        fig_model, axis_model = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig_model.suptitle(title)

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis_model[i, j]
            ax.set_title(f"Reach {k + 1}")

            ax.plot(q_bck_out_ts[k, :], "-b", label="free run")
            ax.plot(q_ana_out_ts[k, :], linestyle="--", color=(1.0, 0.5, 0.), label="analysis step")
            ax.plot(q_final[k, :], "-r", label="analysis run")
            ax.set_ylabel("discharge")
            if bool_diag_obs:
                ax.set_ylabel("height")
            ax.set_ylim((0., flt_max_q))
            ax.set_xlim((0., np.amax(self.vec_t_obs)))

            str_labl_obs = "Q obs"
            if bool_diag_obs:
                str_labl_obs = "H obs"
            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs['reach']}.")
                if row_obs == k:
                    ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label=str_labl_obs)

            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis_model[i, j]
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
    dct_ctl = {"experiment_type": "parameter"}
    my_assim = DAExperiment(forward_model=free_run,
                            dct_obs=dct_obs_1,
                            dct_ctl={"experiment_type": "parameter", "variable": "k"})
    my_assim.sig_o = 0.5
    my_assim.sig_b = 0.25
    my_assim.perform_assim()
    # B_in = np.array([
    #     [SIG_B_Q**2., 0., 0., 0., SIG_B_Q**2.*0.0625],
    #     [0., SIG_B_Q**2., 0., 0., SIG_B_Q**2.*0.125],
    #     [0., 0., SIG_B_Q**2., 0., SIG_B_Q**2.*0.25],
    #     [0., 0., 0., SIG_B_Q**2., SIG_B_Q**2.*0.5],
    #     [SIG_B_Q**2.*0.0625, SIG_B_Q**2.*0.125, SIG_B_Q**2.*0.25, SIG_B_Q**2.*0.5, SIG_B_Q**2.]
    # ])
    # my_assim.perform_assim(b_in=B_in)
