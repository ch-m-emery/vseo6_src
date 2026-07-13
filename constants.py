"""
constants.py module
"""

import numpy as np

# River routing model configuration
N_SIZE_NETWORK = 5
MAT_NETWORK = np.array([[0, 0, 0, 0 ,0],[0, 0, 0, 0 ,0],[1, 1, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 1, 1, 0]])
N_LONGEST_PATH = 2

# rating curve parameter H = a * Q ** b
RC_ALPHA = 0.015
RC_BETA = 1.5

# Truth model parameters
K_TRUTH = 0.75
X_TRUTH = 0.2

# A priori model parameters
K_PRIOR = 0.5
X_PRIOR = 0.3

# Model run parameters
D_T = 0.05
N_ITER = 250
VEC_T = np.arange(start=0, stop=D_T * N_ITER, step=D_T)

# Model inflow
QIN_TS = np.ones((N_SIZE_NETWORK, N_ITER))
QIN_TS[0, :] = 20. + 5. * np.sin(VEC_T)
QIN_TS[3, :] = 5. + 2. * np.cos(VEC_T)

# Model initial condition
Q_T0 = np.array([20., 1., 22., 5., 28.])

# Data assimilation - observation vector
T_OBS_STEP = 25
VEC_TOBS = np.arange(start=T_OBS_STEP, stop=N_ITER+1, step=T_OBS_STEP)
SIG_O_Q = 1.
SIG_O_H = 0.1

# Data assimilation - control vector
SIG_B_Q = 0.5
SIG_B_K = 0.4

