# DeepLearning_FacialRecognition

This repository provides an approach to use deep learning for facial recognition.
I have used VGG-16 pre trained model and then using transfer learning have created model to classsify face.
For ease of usage i have designed a dashboard using Qlikview that would help in
1. Retraining model on new data
2. Testing model on given image
3. Testing model using webcam

The data used is some images downloaded from google and hence I used facial detection algorithm(HAARCascade) to preprocess images.

## Future Work:

Right now it only works with front facing facial images. Future work will include getting dataset to train model to identify sentiment on non-front facing images as well. 
