# ExpressiveCanvas: A Mirror of Emotions in Ever-Shifting Pixels

This repository contains the implementation of the **ExpressiveCanvas** project, which is designed to visually represent human emotions through dynamic and expressive pixel art. The project integrates deep learning models to analyze and interpret facial expressions, transforming these into an ever-changing canvas of pixels that mirror the user's emotions in real-time.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Overview](#technical-overview)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [References](#references)

## Introduction

**ExpressiveCanvas** is an innovative project that bridges the gap between technology and art by transforming facial emotions into visually captivating pixel representations. This project uses deep learning algorithms to detect emotions from facial expressions and translates them into dynamic visual patterns on a digital canvas. It explores the potential of emotion-driven art and its applications in digital media, interactive installations, and more.

## Project Structure

```bash
├── Face_Emotion_Model.py      # Script for training and using the face emotion model
├── IrCode                     # Directory for IR code handling (if applicable)
├── Model                      # Pretrained models and related scripts
├── posenet.py                 # Script for PoseNet implementation (if applicable)
├── Pose_result                # Directory for pose estimation results (if applicable)
├── static                     # Static files (CSS, JavaScript, images)
├── templates                  # HTML templates for the web interface
├── app.py                     # Main application file
├── time_measurements.db       # Database for storing time measurements
├── README.md                  # Project README file
```

## Installation

To run this project locally, follow these steps:

### Clone the repository:

```bash
git clone https://github.com/ishan22399/ExpressiveCanvas-A-Mirror-of-Emotions-in-Ever---Shifting-Pixels.git
cd ExpressiveCanvas-A-Mirror-of-Emotions-in-Ever---Shifting-Pixels
```

### Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the application:

```bash
python app.py
```

## Usage

Once the application is running, you can use the web interface to:

- Capture live video input or upload images.
- Analyze the captured image for emotion detection.
- Generate and view the corresponding pixel art representation in real-time.

### Face Emotion Model

- The model is trained to recognize a variety of emotions such as happiness, sadness, anger, and surprise.
- Emotion recognition results are processed and converted into pixel art on the canvas.

### Pose Estimation (Optional)

- The project includes a script for PoseNet, which can be used for pose estimation if needed.

## Technical Overview

### Methodology

The project uses a Convolutional Neural Network (CNN) to detect and classify emotions from facial images. The detected emotion is then mapped to a predefined set of pixel art patterns, which are displayed on a digital canvas.

### Implementation Details

- **Face Emotion Model:** Trained using labeled emotion datasets.
- **Pixel Art Generator:** Converts detected emotions into corresponding visual patterns.
- **Web Interface:** Built with Flask, allows users to interact with the model through a browser.

## Results

The system successfully maps detected emotions to dynamic pixel art representations. The results show that the model can accurately interpret emotions and generate corresponding visual outputs in real-time, with minimal latency.

## Future Work

Future enhancements may include:

- Integrating more complex pixel art patterns.
- Extending the system to support group emotion detection.
- Improving real-time processing speeds.

## Contributing

Contributions to this project are welcome. Feel free to fork the repository, make your changes, and submit a pull request.

## References

If you use this project in your research, please cite the following paper:

**ExpressiveCanvas: A Mirror of Emotions in Ever-Shifting Pixels**  
*Authors*: Ishan Shivankar, et al.  
Published in *TechRxiv*, 2024.  
[DOI: 10.36227/techrxiv.171925113.31530609/v1](https://doi.org/10.36227/techrxiv.171925113.31530609/v1)


