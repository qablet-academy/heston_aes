"""
Basic Hestom Model with log Euler discretization, without any other optimizations.
"""

import numpy as np
from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter, Forwards
from numpy.random import SFC64, Generator


class HestonMCBasic(MCFixedStep):
    def reset(self):
        """The advance method does the real work of the simulation. The __init__ method
        just makes the necessary parameters handy."""

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

        self.heston_params = (
            self.dataset["HESTON"]["LONG_VAR"],
            self.dataset["HESTON"]["VOL_OF_VOL"],
            self.dataset["HESTON"]["MEANREV"],
            self.dataset["HESTON"]["CORRELATION"],
        )

        # create a random number generator
        self.rng = Generator(SFC64(self.dataset["MC"]["SEED"]))

        # Initialize the processes x (log stock) and v (variance)
        self.x_vec = np.zeros(self.n)  #
        self.v_vec = np.full(self.n, self.dataset["HESTON"]["INITIAL_VAR"])

        self.cur_time = 0

    def step(self, new_time):
        """Update x_vec, v_vec in place when we move simulation by time dt."""
        dt = new_time - self.cur_time

        (theta, vvol, meanrev, corr) = self.heston_params
        fwd_rate = self.asset_fwd.rate(new_time, self.cur_time)

        # generate the wiener processes
        cov = [[dt, corr * dt], [corr * dt, dt]]
        dz_pair = self.rng.multivariate_normal([0, 0], cov, self.n).transpose()

        vol_vec = np.sqrt(np.maximum(0.0, self.v_vec))  # Floor

        # Update log stock process
        self.x_vec += (fwd_rate - vol_vec * vol_vec / 2.0) * dt
        self.x_vec += vol_vec * dz_pair[0]

        # update the variance process
        self.v_vec += (theta - self.v_vec) * meanrev * dt
        self.v_vec += vvol * vol_vec * dz_pair[1]

        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the unit at the current time."""
        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)
        elif unit == "variance":
            return self.v_vec

    def get_df(self):
        return self.discounter.discount(self.cur_time)
