# Cyber-security-in-Power-Grid-using-Deep-Learning
Detecting data spoofing cyber attacks in synchronized measurements from phasor measurement units (PMUs) using deep learning and spatial fingerprint extraction.

This repository contains code and resources for a project focused on detecting data spoofing attacks in synchronized measurements obtained from phasor measurement units (PMUs) in the power industry. The project aims to enhance the security and reliability of power grids by identifying the source of PMU data and detecting potential cyber attacks using deep learning techniques.

## Abstract

The modern power industry heavily relies on innovation and technological advancements to support rapid urbanization and meet the increasing demand for electricity. Synchronized measurements obtained from phasor measurement units (PMUs) have proven beneficial for real-time grid monitoring and control, providing reliable predictions on system security. However, as the use of synchronized measurements grows, so do the cybersecurity concerns surrounding the communication network of synchro-phasors.

This project focuses on extracting spatial fingerprints from synchronized frequency measurements obtained from different PMUs. These fingerprints serve as unique identifiers for the source authentication of PMU data. By building a deep learning classifier, the project aims to detect the presence of data spoofing cyber attacks in PMU data.

The code in this repository consists of three main parts:

1. `generate_fingerprints.py`: This script generates and saves spatial fingerprints of genuine and spoofed PMU data. It includes functions for signal filtering, fingerprint extraction using the Short-Time Fourier Transform (STFT), and data generation for both genuine and spoofed signals.

2. `main.py`: This script trains a deep learning model using the generated spatial fingerprints. It utilizes the TensorFlow library to build a multi-layer MD CNN (Multi-Dilated Convolutional Neural Network) model. The model is trained on the extracted fingerprints and saved as `mdl.h5`.

3. `testing.py`: This script demonstrates the detection capability of the trained model. It loads the saved model and performs inference on new PMU data. The script includes an example of loading an image, preprocessing it, and making predictions using the trained model.

## Usage

To use this project, follow these steps:

1. Run the `generate_fingerprints.py` script to generate spatial fingerprints of genuine and spoofed PMU data. Adjust the parameters and file paths as needed.

2. Execute the `main.py` script to train the deep learning model using the generated fingerprints. The trained model will be saved as `mdl.h5`.

3. Use the `testing.py` script to test the model's detection capability. Adjust the file paths as necessary and provide new PMU data for inference.

Please refer to the code comments and documentation within each script for detailed explanations of the implementation.

## Requirements

The following dependencies are required to run the code:

- Python (>= 3.6)
- TensorFlow (>= 2.0)
- NumPy
- SciPy
- scikit-learn
- PIL

Ensure that these dependencies are installed in your environment before running the code.

## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


## Acknowledgments

The development of this project is inspired by the need for enhanced cybersecurity in the power industry. The project makes use of deep learning techniques to detect data spoofing attacks in synchronized measurements obtained from phasor measurement units (PMUs). Special thanks to the researchers and contributors in the field of power grid security and deep learning for their valuable work and insights.

