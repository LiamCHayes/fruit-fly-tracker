# Self-supervised fruit fly detection

Uses the synthetically generated dataset and self-supervision techniques on background images to build a fruit fly detection model for a single frame. First, train an encoder on the background images using self-supervision (image completeion, rotation prediction, colorization, etc) to understand the environemnt. Using this pre-trained encoder we train a pixel-wise segmentation model for predicting fruit fly location.

TODO list:
- [ ] Write the ResNet encoder
- [ ] Write the self-supervision training loop for the encoder
    - [ ] Decoders for things like image completion and rotation prediction
- [ ] Write the decoder for pixel-wise segmentation
- [ ] Train with self-supervision
- [ ] Fine-tune for pixel-wise segmentation
- [ ] Develop a metric for model performance to use for each iteration (performance on real fruit flies?)

Increase complexity until we can overfit on some sample data:
- [ ] Downsize image to fit within the ResNet image size limits (Not ideal since we probably need the high resolution to detect fruit flies)
- [ ] Adapt ResNet encoder to a Hi-ResNet encoder to handle higher resolution inputs
- [ ] Crop or tile the input so it fits within the size limits of our model
- [ ] Weighted average of loss based on the color of the background around the fly. Darker backgrounds carry less penalty than ligher ones where the flies are more obvious.
- [ ] Self-supervision tasks that attempt to match the image processing techniques that make the flies more obvious

Generalize for real-world performance:
- [ ] Train on more backgrounds
- [ ] Set up the pitch-yaw platform and mount the camera. Have the camera scan the environment and perform self-supervision on these recorded frames. Fine tune with generated data and attempt to detect fruit flies with the encoder trained on the scanned background frames.

Future directions:
- [ ] Adapt for tracking fly trajectory over time
- [ ] Adapt for trajectory prediction
- [ ] Use optical flow and SLAM techniques to detect, track, and predict trajectories while the camera is tilting and panning

