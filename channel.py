r"""
  ____  ____      _    __  __  ____ ___
 |  _ \|  _ \    / \  |  \/  |/ ___/ _ \
 | | | | |_) |  / _ \ | |\/| | |  | | | |
 | |_| |  _ <  / ___ \| |  | | |__| |_| |
 |____/|_| \_\/_/   \_\_|  |_|\____\___/
                           research group
                             dramco.be/

  KU Leuven - Technology Campus Gent,
  Gebroeders De Smetstraat 1,
  B-9000 Gent, Belgium

         File: channel.py
      Created: 2019-09-19
       Author: Gilles Callebaut
      Version: -
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.constants import speed_of_light as c
import cmath
import math


class Channel:

    def __init__(self, num_bs_antennas=32, num_ms_antennas=1, antenna_conf="ULA", num_of_paths=-1, wavelength=-1,
                 d_ms=0.5, d_bs=0.5):
        assert num_ms_antennas == 1, "We expect the MS to only have one antenna (for now)"

        self._num_of_paths = (num_bs_antennas / 2) if num_of_paths == -1 else num_of_paths
        self._num_bs_antennas = num_bs_antennas
        self._num_ms_antennas = num_ms_antennas
        self._antenna_conf = antenna_conf

        self._wavelength = c / 868000000 if wavelength == -1 else wavelength
        self._d_ms = d_ms
        self._d_bs = d_bs

    @staticmethod
    def _generate_alpha_l(n=1):
        return np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=(n, 2)).view(np.complex128)

    def get_steering_vector(self, angle, bs=True):
        assert self._antenna_conf == "ULA", "Other Antenna Configurations not supported"
        k_dh = 2 * math.pi * self._d_bs  # (2pi/lambda) * self._d_bs * lambda = 2 pi * self._d_bs
        return [cmath.exp(i * 1j * k_dh * math.cos(angle)) for i in range(self._num_bs_antennas)]

    def generate_channels(self, aoa_bs: list, aoa_spread):
        """
        :param aoa_bs: Angle of Arrivals per user at the BS or dominant directions of the users
        :param aoa_spread: Angular Spread of the AoA samples (sigma^2)
        :return: channel instances per antenna (complex-valued array 1xM)

        For each path L an angle is computed where angle ~ N(mu=AoA, sigma2=aoa_spread)

        Transmit power is not included in the channel
        """
        for angle in aoa_bs:
            # holder for the complex channel per antenna
            h = [0] * self._num_bs_antennas
            # sum steering vectors and different paths
            # TODO check angle/angular spread
            for l in range(0, self._num_of_paths):
                angle_for_l = np.random.normal(loc=angle, scale=np.sqrt(aoa_spread))
                h = h + Channel._generate_alpha_l() * self.get_steering_vector(angle_for_l)
