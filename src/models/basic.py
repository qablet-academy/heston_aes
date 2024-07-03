"""
Basic Hestom Model without any optimizations
"""

import numpy as np

from numpy.random import SFC64, Generator

from qablet.base.mc import MCModel, MCStateBase
from qablet.base.utils import Forwards


class HestonMCState(MCStateBase):
    def __init__(self, timetable, dataset):
        """The advance method does the real work of the simulation. The __init__ method
        just makes the necessary parameters handy."""

        super().__init__(timetable, dataset)

        self.n = dataset["MC"]["PATHS"]

        # create a random number generator
        self.rng = Generator(SFC64(dataset["MC"]["SEED"]))

        self.asset = dataset["HESTON"]["ASSET"]
        self.asset_fwd = Forwards(dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)

        self.heston_params = (
            dataset["HESTON"]["LONG_VAR"],
            dataset["HESTON"]["VOL_OF_VAR"],
            dataset["HESTON"]["MEANREV"],
            dataset["HESTON"]["CORRELATION"],
        )

        # Initialize the processes x (log stock) and v (variance)
        self.x_vec = np.zeros(self.n)  #
        self.v_vec = np.full(self.n, dataset["HESTON"]["INITIAL_VAR"])

        self.cur_time = 0

    def advance(self, new_time):
        """Update x_vec, v_vec in place when we move simulation by time dt."""
        dt = new_time - self.cur_time
        if dt < 1e-10:
            return

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
        else:
            return None


class HestonMCModel(MCModel):
    def state_class(self):
        return HestonMCState
