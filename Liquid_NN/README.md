# Liquid Neural Network Testbed

This directory contains a small, self‑contained implementation of a Liquid
Neural Network layer written with NumPy.  It serves as a starting point for
exploring liquid neural networks within the broader bio‑inspired SLAM
project.

## Files

- `liquid_layer.py` – definition of the `LiquidLayer` class.
- `demo.py` – simple script that instantiates a layer and runs it on random
  data to verify it works.

## Running the demo

```bash
python demo.py
```

The script will print the shape of the produced sequence and the final hidden
state of the network.
