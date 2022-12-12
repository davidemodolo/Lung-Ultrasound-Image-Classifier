# Lung Ultrasound Image classifier

As explained in this [paper](https://ieeexplore.ieee.org/document/9093068), images are scored as:

0. no artifact in the picture

1. at least one vertical artifact (B-line)

2. small consolidation below the pleural surface

3. wider hyperechogenic area below the pleural surface (> 50%)

We have given a partial dataset from the San Matteo hospital, consisting of 11 patients for a total of ~47k frames.

The model I'm trying to build here is composed by two parts:

- a fine-tuned pre-trained model (resnet18)

- a near duplicate image search module

The idea is to take prediction in which the model is not confident enough and compare the frame to already known frames to (hopefully) enhance the accuracy.

While the project proceeds, I will also update this README file.
