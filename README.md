# VIETNAMESE SIGN LANGUAGE RECOGNITION
A system for recognizing Vietnamese Sign Language using deep learning and computer vision techniques, tailored specifically for Vietnamese sign language.
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Training from Scratch](#training-from-scratch)
## Overview
The Vietnamese Sign Language Recognition system leverages deep learning models and computer vision to interpret Vietnamese sign language gestures. It uses MediaPipe for landmark detection, TensorFlow for model training, and React for a user-friendly interface. The system supports recognition through video files or live webcam feeds.
## Features
- Automated Video Download: Automatically downloads videos for training data.
- Data Preprocessing: Processes and augments data for model training.
- Sign Language Recognition: Recognizes Vietnamese sign language gestures via video or webcam input.
- User Interface: Provides a React-based web interface for easy interaction.
## Requirements
- **Software**:
    - Python 3.8 or higher
    - TensorFlow 2.x
    - Scikit-learn
    - MediaPipe
    - OpenCV
    - Node.js and npm
- **Hardware**:
    - Webcam (required for webcam recognition)
    - GPU (recommended for model training)
## Installation
### 1. Clone the repository
```bash
git clone https://github.com/photienanh/Vietnamese-Sign-Language-Recognition
cd Vietnamese-Sign-Language-Recognition
```
Alternatively, download the ZIP file from GitHub and extract it.
### 2. Install Dependencies
Ensure Python is installed. If not, you can download and install it from the official [Python website](https://www.python.org/downloads/). Then, install the required libraries:
```bash
pip install -r requirements.txt
```
You will also need to install Node.js and npm. You can download them from the official [Node.js website](https://nodejs.org/).
## Usage
The system can be used either by running the pre-trained model or by training a new model from scratch.
### Running the Application
1.  **Install frontend dependencies:**
    ```bash
    cd frontend
    npm install
    ```
2.  **Build the frontend:**
    ```bash
    npm run build
    ```
3.  **Run the backend server:**
    From the root directory of the project:
    ```bash
    uvicorn api:app --reload
    ```
This launches a web server on `http://localhost:8000`. You can access the application by opening this URL in your browser.

### Training from Scratch
To train a new model, follow these steps:
1. Clear Previous Data (optional).
```bash
Get-ChildItem -Path "./" -Directory | Remove-Item -Recurse -Force
```
2. Download Training Data.
```bash
python download_data.py
```

3. Process Data.
```bash
python create_data_augment.py
```

4. Train the Model.
- Open ```training.ipynb``` in a Jupyter Notebook environment.
- Run all cells to train the model.
- Note: Training is computationally intensive and best performed on a GPU-enabled device.
