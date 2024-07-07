"""
Qablet model based on Heston Almost Exact Simulation.
Is is adapted from the implementation by Nicholas Burgess in
(https://github.com/nburgessx/Papers/tree/main/HestonSimulation)
"""

import numpy as np
from numpy.random import SFC64, Generator
from qablet.base.mc import MCModel, MCStateBase
from qablet.base.utils import Forwards


def CIR_Sample(rng, n, kappa, gamma, vbar, s, t, v_s):
    delta = 4.0 * kappa * vbar / gamma / gamma
    c = 1.0 / (4.0 * kappa) * gamma * gamma * (1.0 - np.exp(-kappa * (t - s)))
    kappaBar = (
        4.0
        * kappa
        * v_s
        * np.exp(-kappa * (t - s))
        / (gamma * gamma * (1.0 - np.exp(-kappa * (t - s))))
    )
    sample = c * rng.noncentral_chisquare(delta, kappaBar, n)
    return sample


class HestonAESMCState(MCStateBase):
    def __init__(self, timetable, dataset):
        """Initialize internal state of the model."""
        super().__init__(timetable, dataset)

        # fetch the model parameters from the dataset
        self.n = dataset["MC"]["PATHS"]
        # asset information
        self.asset = dataset["HESTON"]["ASSET"]
        self.asset_fwd = Forwards(dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)

        self.gamma = dataset["HESTON"]["VOL_OF_VAR"]
        self.kappa = dataset["HESTON"]["MEANREV"]
        self.vbar = dataset["HESTON"]["LONG_VAR"]
        self.rho = dataset["HESTON"]["CORRELATION"]

        # create a random number generator
        self.rng = Generator(SFC64(dataset["MC"]["SEED"]))

        # Initialize the arrays
        self.x_vec = np.zeros(self.n)  # process x (log stock)
        # Initialize as a scalar, it will become a vector in the advance method
        self.v = dataset["HESTON"]["INITIAL_VAR"]

        self.cur_time = 0

    def advance(self, new_time):
        """Advance the internal state of the model to current time."""

        dt = new_time - self.cur_time
        if dt < 1e-10:
            return

        r = self.asset_fwd.rate(new_time, self.cur_time)

        # Generate the Brownian Increments
        dw_vec = self.rng.standard_normal(self.n) * np.sqrt(dt)  # * self.vol

        # Exact samples for the variance process
        new_v = CIR_Sample(
            self.rng, self.n, self.kappa, self.gamma, self.vbar, 0, dt, self.v
        )

        # AES Constant Terms
        k0 = (r - self.rho / self.gamma * self.kappa * self.vbar) * dt
        k1 = (
            self.rho * self.kappa / self.gamma - 0.5
        ) * dt - self.rho / self.gamma
        k2 = self.rho / self.gamma

        # Almost Exact Simulation for Log-Normal Asset Process
        self.x_vec += (
            k0
            + k1 * self.v
            + k2 * new_v
            + np.sqrt((1.0 - self.rho**2) * self.v) * dw_vec
        )

        self.v = new_v
        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the unit at the current time."""
        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)
        else:
            return None


class HestonAESMC(MCModel):
    def state_class(self):
        return HestonAESMCState
