"""Discrete held-input transition for the repository spring and DC motor examples."""
import numpy as np


def step(x, u, d, dt, parameters):
    return (np.asarray(parameters["A"]) @ x + np.asarray(parameters["B"]) @ u
            + np.asarray(parameters["E"]) @ d)
