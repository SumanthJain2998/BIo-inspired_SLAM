"""Simple implementation of a Liquid Neural Network layer using NumPy.

This module provides the :class:`LiquidLayer`, a small liquid neural network
layer that integrates a continuous-time differential equation using a simple
Euler solver.  The implementation is intentionally lightweight and
self-contained so it can act as a starting point for experiments.
"""

from __future__ import annotations

import numpy as np


class LiquidLayer:
    """A minimal Liquid Neural Network layer.

    Parameters
    ----------
    input_size: int
        Number of input features.
    hidden_size: int
        Number of neurons in the liquid layer.
    dt: float, optional
        Integration time step.
    seed: int, optional
        Random seed for reproducibility.
    """

    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.01, *, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        # System matrices
        self.A = rng.standard_normal((hidden_size, hidden_size)) * 0.1
        self.W = rng.standard_normal((hidden_size, hidden_size)) * 0.1
        self.U = rng.standard_normal((hidden_size, input_size)) * 0.1
        self.bias = np.zeros(hidden_size)

        # Initial state of the network
        self.state = np.zeros(hidden_size)

    def reset_state(self) -> None:
        """Reset the internal state of the layer to zeros."""
        self.state = np.zeros(self.hidden_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagate a sequence of inputs through the liquid layer.

        Parameters
        ----------
        inputs: np.ndarray, shape (T, input_size)
            Sequence of input vectors.

        Returns
        -------
        np.ndarray, shape (T, hidden_size)
            Sequence of hidden states produced by the network.
        """

        outputs = []
        for u in inputs:
            dx = -self.A @ self.state + np.tanh(self.W @ self.state + self.U @ u + self.bias)
            self.state = self.state + self.dt * dx
            outputs.append(self.state.copy())
        return np.vstack(outputs)
