# Fruit Fly Tracker

**data_generation** generates synthetic data by overlaying black squares on images. Use to train a detection/classification neural network for real fruit flies.

**self_supervision_detection** trains a detection model for a single frame of fruit flies with the synthetic data.

**utilities** contains helper functions for use throughout the repository.

**setup.py** Allows us to `pip install -e .` utilities and other folders that we decide to make modules for use in other code.

Future:
- Extend **self_supervision_detection** to track flies over time
- Predict fly trajectories
- Use SLAM and optical flow in combination with the trained models to detect, track, and predict fruit flies while the camera is moving
- Use 3D reconstruction to judge distances and determine what is in range
- Synchronize trigger with predictions
