# Event-Based Liquid SLAM

> **Simultaneous Localization and Mapping using Liquid Neural Networks and Event Cameras**

This repository implements a novel SLAM pipeline that fuses **event camera data** with **Liquid Neural Networks (LNNs)** — specifically Closed-Form Continuous-Time (CfC) networks wired via Neural Circuit Policies (NCP). Rather than accumulating events into synthetic frames, this system processes the raw asynchronous event stream directly, enabling a fundamentally more natural and efficient representation of spatiotemporal data.

The architecture is inspired by the rodent hippocampal navigation system (RatSLAM), re-implemented using biologically-plausible liquid dynamics. This is an active research project; a journal paper is in preparation.

> **Sister Repository:** The RatSLAM baseline and direct event integration experiments are developed in [`syedamansohrab/Ratslam`](https://github.com/syedamansohrab/Ratslam). See [Linking the Repositories](#linking-the-repositories) below.

---

## Motivation

Standard SLAM systems — and most neural approaches — are built around a frame-based abstraction. Event cameras, however, do not produce frames. They output a continuous stream of asynchronous pixel-level brightness changes at microsecond resolution. Forcing this data into frames discards its primary advantage: temporal precision.

Liquid Neural Networks are continuous-time recurrent networks whose time constants adapt dynamically to the input signal. This makes them a natural fit for the variable-rate event stream. This project explores that match at the systems level.

---

## Core Architecture

The system is a bio-inspired SLAM pipeline with three interacting components:

### 1. Liquid View Cells (LVC) — Place Recognition
A CfC network wired via NCP topology. Sensory neurons receive raw event embeddings; interneurons integrate temporal context; command neurons output a compact view template used for place recognition and loop closure.

### 2. Liquid Pose Cells (LPC) — Continuous Attractor Network
Replaces the discrete competitive attractor of standard RatSLAM with a **Liquid Continuous Attractor Network**. The key property is that the network's time constant adapts to the incoming event rate — when the robot is stationary (sparse events), the pose estimate freezes, eliminating drift. When the robot moves (dense events), the attractor updates rapidly.

### 3. Experience Map
A lightweight topological-metric graph whose nodes store (pose, view template) pairs. Loop closure is triggered when the current view template matches a stored one above a confidence threshold.

---

## Repository Structure

```
Liquid_SLAM/
├── src/
│   ├── core/
│   │   ├── cfc_modified.py          # CfC cell implementation
│   │   ├── ncp_wiring.py            # NCPWiring and NCPCell
│   │   └── liquid_attractor.py      # LiquidAttractor2D (Liquid Pose Cells)
│   ├── data/
│   │   └── event_dataset.py         # EventEncoder and DSEC dataset loader
│   └── slam/
│       └── bio_slam.py              # EventBioSLAM — top-level integration module
├── external/
│   └── ratslam/                     # Git submodule → syedamansohrab/Ratslam
├── test/
│   └── test_liquid_attractor.py     # Unit tests for attractor properties
├── train_dsec_cfc.py                # Training entry point on DSEC dataset
└── README.md
```

---

## Key Components

### `LiquidAttractor2D` (`src/core/liquid_attractor.py`)
The core novelty of this project. Implements a 2D continuous attractor with liquid (adaptive) time constants. Validated independently via unit tests before integration with the event stream.

### `EventBioSLAM` (`src/slam/bio_slam.py`)
Integrates all components into a stateful forward loop. Processes a continuous stream of raw events `(x, y, p, dt)` and maintains running pose and view state across the entire sequence.

### `EventEncoder` (`src/data/event_dataset.py`)
Embeds raw events into dense vectors suitable for driving the CfC. Handles polarity-aware aggregation and spatial binning.

---

## Installation

```bash
git clone --recurse-submodules https://github.com/<your-username>/Liquid_SLAM.git
cd Liquid_SLAM
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 2.0, NumPy, Matplotlib, pytest

---

## Running the Tests

The attractor module has a standalone test suite that verifies its core properties before integration with the full SLAM pipeline.

```bash
python test/test_liquid_attractor.py
```

| Test | Property Verified |
|---|---|
| `test_static_stability` | Bump remains stable with zero velocity input |
| `test_liquid_time_constant` | High event rate updates state; zero rate freezes it |
| `test_path_integration` | Bump displacement is proportional to integrated velocity |

---

## Training on DSEC

```bash
python train_dsec_cfc.py --config configs/dsec_cfc.yaml
```

The training loop is stateful: the hidden state is carried across the entire event sequence, with the inter-event interval `dt` passed directly to the liquid dynamics.

---

## Linking the Repositories

The RatSLAM baseline and direct event integration work lives in [`syedamansohrab/Ratslam`](https://github.com/syedamansohrab/Ratslam). It is tracked here as a Git submodule under `external/ratslam/`.

**To add the submodule (first time setup):**
```bash
git submodule add https://github.com/syedamansohrab/Ratslam.git external/ratslam
git commit -m "Add Ratslam as submodule"
git push
```

**To clone this repo with the submodule included:**
```bash
git clone --recurse-submodules https://github.com/<your-username>/Liquid_SLAM.git
```

**To update the submodule to the latest commit:**
```bash
git submodule update --remote external/ratslam
git commit -m "Update Ratslam submodule"
```

---

## Related Work

- **Lechner et al. (2020)** — *Neural circuit policies enabling auditable autonomy*. Nature Machine Intelligence.
- **Hasani et al. (2021)** — *Closed-form continuous-time neural networks*. NeurIPS.
- **Chahine et al. (2023)** — *Towards autonomous systems with liquid neural networks*. Science Robotics.
- **Hasani et al. (2022)** — *Liquid time-constant networks*. AAAI.

---

## Current Status

| Module | Status |
|---|---|
| `CfCCell` / `NCPWiring` | 🔧 In progress |
| `EventEncoder` | 🔧 In progress |
| `LiquidAttractor2D` | 🔧 In progress |
| `EventBioSLAM` integration | 🔧 In progress |
| Loop closure with LVC | 📋 Planned |
| Full DSEC benchmark | 📋 Planned |

---

## License

MIT
