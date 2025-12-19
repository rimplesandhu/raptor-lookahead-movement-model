# Raptor Look-Ahead Movement Model

A data-driven Markov model for predicting fine-scale, 3D flight movements of soaring raptors using look-ahead environmental factors.

## Overview

This model predicts low-altitude flight behavior of golden eagles at rotor-swept altitudes (≤200 m AGL) for up to 3 minutes. Unlike traditional movement models, it incorporates directionally-explicit environmental conditions within an eagle's line of sight to capture how raptors make directional decisions based on terrain and updraft conditions ahead of them.

**Key features:**
- 1-second temporal resolution
- Predicts vertical speed, horizontal speed, and heading rate
- Uses readily available environmental data (elevation, wind)
- Calibrated on telemetry data from eastern and western USA
- Applicable to turbine curtailment and conservation planning

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/rimplesandhu/raptor-lookahead-movement-model.git
cd raptor-lookahead-movement-model
uv sync
```
The python scripts and figure generating jupyter notebooks are in the examples folder. Note that the data required for running those scripts is not provided here since the golden eagle telemetry data is sensitive and not available to public.

## Model Structure

The model uses a discrete-time first-order Markov process with an empirical relationship between heading rate (ω) and orographic updraft gradients:

```
ω = c_ω * (Δw_o^(α,d)) / (d · α)
```

where α is the look-ahead angle, d is the look-ahead distance, and Δw_o is the updraft difference.

## Performance

Validated on 200 three-minute tracks:
- **Windy + RSZ conditions:** Predicts location within ~6 rotor diameters at 3 minutes
- **Windy + RSZ + Soaring:** Within ~4 rotor diameters at 3 minutes
- **2-3x improvement** over constant velocity baseline

## Citation

```
Sandhu, R., Tripp, C., Quon, E., Thedin, R., Lanzone, M., Braham, M.A., 
Miller, T.A., Farmer, C.J., Brandes, D., Katzner, T. (2025). 
Movement models to predict low-altitude flight of soaring birds using 
look-ahead environmental factors. Ecology and Evolution.
```

## License

BSD 3-Clause License

## Contact

Rimple Sandhu - Rimple.Sandhu@nrel.gov
