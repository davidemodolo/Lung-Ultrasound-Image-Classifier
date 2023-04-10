# Lung Ultrasound Images Classifier

This project aims to find an alternative way to classify Lung Ultrasound Images of patients affected by COVID-19.

This repository is a student project for the _Medical Imaging Diagnostic_ course of the Master's Degree in Artificial Intelligent Systems at the University of Trento, a.y. 2022-2023.

## Before starting

As explained in the paper [Deep Learning for Classification and Localization of COVID-19 Markers in Point-of-Care Lung Ultrasound](https://ieeexplore.ieee.org/document/9093068), images are scored as:

0. no artifact in the picture

1. at least one vertical artifact (B-line)

2. small consolidation below the pleural surface

3. wider hyperechogenic area below the pleural surface (> 50%)

We have been given a partial dataset from the San Matteo hospital, consisting of 11 patients for a total of ~47k frames.

## My approach

The model I'm trying to build here is composed by three main parts:

- a fine-tuned **pre-trained model** fitted on this problem;

- a **binary classifier** that tries to predict, from the first model's behaviour, if it is confident enough; if `True`, the prediction is definitive, if `False`, the model proceeds to the next part;

- a **similarity model** to retrieve the similarity between the input frame and the training frames (probably [t-SNE](https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42#:~:text=t%2DSNE%20is%20a%20powerful,parameters%20that%20can%20be%20tweaked)).

The idea is to take prediction in which the model is not confident enough and compare the frame to already known frames to (hopefully) enhance the accuracy.

## In depth

### PRE-TRAINED MODEL

ResNet18

VGG16

Squeezenet1

### BINARY CLASSIFIER

SVC

4 SVCs

Deep model on both resnet and squeezenet results

### SIMILARITY MODEL

Near duplicate image search 

t-SNE using model features

t-SNE using resnet embeddings

t-SNE using "raw" images
