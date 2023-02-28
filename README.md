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
- [ ] transformations from professor's papers for data augmentation
- [ ] use different pre-trained models
- [ ] use different similarity models/modality

Try using padding and resizing

Also classificator softmax -> correct/wrong prediction