import numpy as np
from math import sqrt
from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter, Forwards
from numpy.random import SFC64, Generator

def CIR_Sample(rng, shape, kappa, gamma, vbar, s, t, v_s):
    delta = 4.0 * kappa * vbar / gamma / gamma
    c = 1.0 / (4.0 * kappa) * gamma * gamma * (1.0 - np.exp(-kappa * (t - s)))
    kappaBar = (
        4.0
        * kappa
        * v_s
        * np.exp(-kappa * (t - s))
        / (gamma * gamma * (1.0 - np.exp(-kappa * (t - s))))
    )
    sample = c * rng.noncentral_chisquare(delta, kappaBar, shape)
    return sample

class HestonBestMC(MCFixedStep):
    def reset(self):
        """Initialize internal state of the model."""
        # fetch the model parameters from the dataset
        self.shape = self.dataset["MC"]["PATHS"]
        self.n = self.shape >> 1
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
        self.x_vec = np.zeros(self.shape)  # process x (log stock)
        # Initialize as a scalar, it will become a vector in the advance method
        self.v = self.dataset["HESTON"]["INITIAL_VAR"]
        self.v_vec = np.full(self.shape, self.dataset["HESTON"]["INITIAL_VAR"])

        self.cur_time = 0
        self.tmp_vec = np.empty(self.shape, dtype=np.float64)
        self.dz1_vec = np.empty(self.shape, dtype=np.float64)
        self.dz2_vec = np.empty(self.shape, dtype=np.float64)
        self.vol_vec = np.empty(self.shape, dtype=np.float64)
        self.vplus_vec = np.empty(self.shape, dtype=np.float64)

    def step(self, new_time):
        """Advance the internal state of the model to current time."""
        dt = new_time - self.cur_time

        r = self.asset_fwd.rate(new_time, self.cur_time)
        sqrtdt= sqrt(dt)

        # Generate the Brownian Increments
        dw_vec = self.rng.standard_normal(self.shape) * np.sqrt(dt)

        # Exact samples for the variance process using CIR model
        new_v = CIR_Sample(
            self.rng, self.shape, self.kappa, self.gamma, self.vbar, 0, dt, self.v
        )

        # AES Constant Terms for the stock process
        k0 = (r - self.rho / self.gamma * self.kappa * self.vbar) * dt
        k1 = (
            self.rho * self.kappa / self.gamma - 0.5
        ) * dt - self.rho / self.gamma
        k2 = self.rho / self.gamma

        # Almost Exact Simulation for Log-Normal Asset Process (AES)
        self.x_vec += (
            k0
            + k1 * self.v
            + k2 * new_v
            + np.sqrt((1.0 - self.rho**2) * self.v) * dw_vec
        )
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
            sqrtdt * sqrt(1 - self.rho * self.rho),
            self.dz2_vec[0:n],
            out=self.dz2_vec[0:n],
        )
        np.negative(self.dz2_vec[0:n], out=self.dz2_vec[n:])
        np.multiply(self.rho, self.dz1_vec, out=self.tmp_vec)  # second term
        np.add(self.dz2_vec, self.tmp_vec, out=self.dz2_vec)

        # vol = sqrt(max(v, 0))
        np.maximum(0.0, self.v_vec, out=self.vplus_vec)
        np.sqrt(self.vplus_vec, out=self.vol_vec)

        # update the current value of v (variance process)
        # first term: v += meanrev * (theta - v) * dt
        np.subtract(self.vbar, self.v_vec, out=self.tmp_vec)
        np.multiply(self.tmp_vec, (self.kappa * dt), out=self.tmp_vec)
        np.add(self.v_vec, self.tmp_vec, out=self.v_vec)

        # second term: v += vvol * vol * dz2
        np.multiply(self.gamma, self.vol_vec, out=self.tmp_vec)
        np.multiply(self.tmp_vec, self.dz2_vec, out=self.tmp_vec)
        np.add(self.v_vec, self.tmp_vec, out=self.v_vec)

        # Milstein correction
        # third term: v += 0.25 * vvol * vvol * (dz2 ** 2 - dt)
        np.multiply(self.dz2_vec, self.dz2_vec, out=self.tmp_vec)
        np.subtract(self.tmp_vec, dt, out=self.tmp_vec)
        np.multiply(
                0.25 * self.gamma * self.gamma,
                self.tmp_vec,
                out=self.tmp_vec,
            )
        np.add(self.v_vec, self.tmp_vec, out=self.v_vec)

        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the unit at the current time."""
        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)
        elif unit == "variance":
            return self.v_vec

    def get_df(self):
        return self.discounter.discount(self.cur_time)
