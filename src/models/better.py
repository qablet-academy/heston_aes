"""
Heston Model with Milstein correction and antithetic variates.
It also optimizes the cost of numpy operations.
"""

from math import sqrt

import numpy as np
from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter, Forwards
from numpy.random import SFC64, Generator


# Define a class for the state of a single asset Heston MC process
class HestonMCBetter(MCFixedStep):
    milstein = False

    def reset(self):
        self.shape = self.dataset["MC"]["PATHS"]
        self.timestep = self.dataset["MC"]["TIMESTEP"]

        assert self.shape % 2 == 0, "Number of paths must be even"
        self.n = self.shape >> 1  # divide by 2

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
        self.x_vec = np.zeros(self.shape)
        self.v_vec = np.full(self.shape, self.dataset["HESTON"]["INITIAL_VAR"])

        # We will reduce time spent in memory allocation by creating arrays in advance
        # and reusing them in the `advance` function which is called repeatedly.
        # though the values from one timestep are not reused in the next.
        self.tmp_vec = np.empty(self.shape, dtype=np.float64)
        self.dz1_vec = np.empty(self.shape, dtype=np.float64)
        self.dz2_vec = np.empty(self.shape, dtype=np.float64)
        self.vol_vec = np.empty(self.shape, dtype=np.float64)
        self.vplus_vec = np.empty(self.shape, dtype=np.float64)

        self.cur_time = 0

    def step(self, new_time):
        """Update x_vec, v_vec in place when we move simulation by time dt."""
        dt = new_time - self.cur_time

        (theta, vvol, meanrev, corr) = self.heston_params
        fwd_rate = self.asset_fwd.rate(new_time, self.cur_time)

        sqrtdt = sqrt(dt)
        n = self.n

        # To improve preformance we will break up the operations into np.multiply,
        # np.add, etc. and use the `out` parameter to avoid creating temporary arrays.

        # generate the random numbers, using antithetic variates
        # we calculate dz1 = normal(0,1) * sqrtdt
        self.rng.standard_normal(n, out=self.dz1_vec[0:n])
        np.multiply(sqrtdt, self.dz1_vec[0:n], out=self.dz1_vec[0:n])
        np.negative(self.dz1_vec[0:n], out=self.dz1_vec[n:])

        # we calculate dz2 = normal(0,1) * sqrtdt * sqrt(1 - corr * corr) + corr * dz1
        self.rng.standard_normal(n, out=self.dz2_vec[0:n])
        np.multiply(
            sqrtdt * sqrt(1 - corr * corr),
            self.dz2_vec[0:n],
            out=self.dz2_vec[0:n],
        )
        np.negative(self.dz2_vec[0:n], out=self.dz2_vec[n:])
        np.multiply(corr, self.dz1_vec, out=self.tmp_vec)  # second term
        np.add(self.dz2_vec, self.tmp_vec, out=self.dz2_vec)

        # vol = sqrt(max(v, 0))
        np.maximum(0.0, self.v_vec, out=self.vplus_vec)
        np.sqrt(self.vplus_vec, out=self.vol_vec)

        # update the current value of x (log Stock process)
        # first term: x += (fwd_rate - vol * vol / 2.) * dt
        np.divide(self.vplus_vec, 2, out=self.tmp_vec)
        np.subtract(fwd_rate, self.tmp_vec, out=self.tmp_vec)
        np.multiply(self.tmp_vec, dt, out=self.tmp_vec)
        np.add(self.x_vec, self.tmp_vec, out=self.x_vec)

        # second term: x += vol * dz1
        np.multiply(self.vol_vec, self.dz1_vec, out=self.tmp_vec)
        np.add(self.x_vec, self.tmp_vec, out=self.x_vec)

        # update the current value of v (variance process)
        # first term: v += meanrev * (theta - v) * dt
        if self.milstein:
            np.subtract(theta, self.vplus_vec, out=self.tmp_vec)
        else:
            np.subtract(theta, self.v_vec, out=self.tmp_vec)
        np.multiply(self.tmp_vec, (meanrev * dt), out=self.tmp_vec)
        np.add(self.v_vec, self.tmp_vec, out=self.v_vec)

        # second term: v += vvol * vol * dz2
        np.multiply(vvol, self.vol_vec, out=self.tmp_vec)
        np.multiply(self.tmp_vec, self.dz2_vec, out=self.tmp_vec)
        np.add(self.v_vec, self.tmp_vec, out=self.v_vec)

        # Milstein correction
        # third term: v += 0.25 * vvol * vvol * (dz2 ** 2 - dt)
        if self.milstein:
            np.multiply(self.dz2_vec, self.dz2_vec, out=self.tmp_vec)
            np.subtract(self.tmp_vec, dt, out=self.tmp_vec)
            np.multiply(
                0.25 * vvol * vvol,
                self.tmp_vec,
                out=self.tmp_vec,
            )
            np.add(self.v_vec, self.tmp_vec, out=self.v_vec)
            # print(f"M {self.tmp_vec.mean() * 1000_000:.4f}")
        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the unit at the current time."""
        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)
        elif unit == "variance":
            return self.v_vec

    def get_df(self):
        return self.discounter.discount(self.cur_time)


class HestonMCMilstein(HestonMCBetter):
    milstein = True
