# Lung Ultrasound Image classifier

As explained in this [paper](https://ieeexplore.ieee.org/document/9093068), images are scored as:

0. no artifact in the picture

1. at least one vertical artifact (B-line)

2. small consolidation below the pleural surface

3. wider hyperechogenic area below the pleural surface (> 50%)

We have given a partial dataset from the San Matteo hospital, consisting of 11 patients for a total of ~47k frames.

The model I'm trying to build here is composed by two parts:

- a fine-tuned pre-trained model (resnet18)
    - first k fold 8-3 5epochs to find the best configuration of train/test patients
    - choose the best configuration and start from there for augmentation used in paper **Deep Learning for Classification and Localization of COVID-19 Markers in Point-of-Care Lung Ultrasound**

- a near duplicate image search module OR [t-SNE](https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42#:~:text=t%2DSNE%20is%20a%20powerful,parameters%20that%20can%20be%20tweaked).

The idea is to take prediction in which the model is not confident enough and compare the frame to already known frames to (hopefully) enhance the accuracy.

After training the pre-trained network, the whole dataset is checked in the network. If we get a wrong result, save softmax value. Learn if there is some correlation between error numbers and confidence. Then, test again and if any "error" behaviour is found, we pass the image into the second network to find the final result.

My todo:
- [x] **BALANCE TRAIN CORR == 1**
- [x] function to compute mean and sd of a dataset
- [x] compute mean and sd con the RGB images
- [x] plot the number of frames by score
- [x] convert all images to `tf.train.Example` **TOO LARGE, keep png_s**
- [x] transformations from professor's papers for data augmentation
- [x] during train each frame is passed clean, and 2 times transformed. A specific patient is kept out of everything for the final test: choose the one to leave out
    patient 1047 has 1151 frames, patient 1051 has 1239 frames. Those two are the patient with less frames, **1051** is more linear between the scores, better one for the test
- [x] use different pre-trained models
- [ ] use different similarity models/modality
- [x] confusion matrix for the first model

Steps for presentation:
- choose the best combination of patients based on the std
- train ResNet18
- prepare the second
- learn when to ask the second
- performance only with the first
- performance only on the second
- total performance
- example for retrieving

Also classificator softmax -> correct/wrong prediction
Add in report little explaination of t-sne and parameters