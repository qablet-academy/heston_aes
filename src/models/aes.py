"""
Qablet model based on Heston Almost Exact Simulation.
Is is adapted from the implementation by Nicholas Burgess in
(https://github.com/nburgessx/Papers/tree/main/HestonSimulation)
"""

import numpy as np
from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter, Forwards
from numpy.random import SFC64, Generator


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


class HestonAESMC(MCFixedStep):
    def reset(self):
        """Initialize internal state of the model."""

        # fetch the model parameters from the dataset
        self.n = self.dataset["MC"]["PATHS"]
        self.timestep = self.dataset["MC"]["TIMESTEP"]

        # asset information
        self.asset = self.dataset["HESTON"]["ASSET"]
        self.asset_fwd = Forwards(self.dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)
        self.discounter = Discounter(
            self.dataset["ASSETS"][self.dataset["BASE"]]
        )

        self.gamma = self.dataset["HESTON"]["VOL_OF_VOL"]
        self.kappa = self.dataset["HESTON"]["MEANREV"]
        self.vbar = self.dataset["HESTON"]["LONG_VAR"]
        self.rho = self.dataset["HESTON"]["CORRELATION"]

        # create a random number generator
        self.rng = Generator(SFC64(self.dataset["MC"]["SEED"]))

        # Initialize the arrays
        self.x_vec = np.zeros(self.n)  # process x (log stock)
        # Initialize as a scalar, it will become a vector in the advance method
        self.v = self.dataset["HESTON"]["INITIAL_VAR"]

        self.cur_time = 0

    def step(self, new_time):
        """Advance the internal state of the model to current time."""

        dt = new_time - self.cur_time

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
        elif unit == "variance":
            return self.v

    def get_df(self):
        return self.discounter.discount(self.cur_time)
