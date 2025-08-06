"""Run a small demonstration of the :class:`LiquidLayer`."""

import numpy as np
from liquid_layer import LiquidLayer


def main() -> None:
    # Generate a random input sequence of length 100 with 3 features
    inputs = np.random.randn(100, 3)

    # Create and run the liquid layer
    layer = LiquidLayer(input_size=3, hidden_size=5, dt=0.05, seed=0)
    outputs = layer.forward(inputs)

    print("Output shape:", outputs.shape)
    print("Final state:", outputs[-1])


if __name__ == "__main__":
    main()
