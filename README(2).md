
# Multiclassification for four classes of skin lessions





## Acknowledgements

 - [the dataset](https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset)
 - [the research paper of the dataset ](https://www.sciencedirect.com/science/article/pii/S0893608023000850#:~:text=Furthermore%2C%20we%20proposed%20and%20evaluated,93.19%25%20and%2098.91%25%20respectively.)

# 1. Exploring the dataset

This dataset consists of four classes: Monkeypox, Chickenpox, Measles, and Normal. All the image classes are collected from internet-based sources. The entire dataset has been developed by the Department of Computer Science and Engineering, Islamic University, Kushtia-7003, Bangladesh.
- all the image in the dataset shape was (224,224,3)
- the dataset is unbalanced, Measles and Chickenpox are considered minority classes so for the next step is Data Augmentation
<img src="https://github.com/kholoudabdelmohsen7/Multiclassification-for-four-classes-of-skin-lessions/blob/b65047f8bf5f84d3f23c58acbf3ac3835f3aceff/unbalanced.png">

# 2. Data Augmentation
For the Geometric Transformations

- Rotation Range was set to 30 degrees
- Heigh shift range, shear range and zoom range was set to 0.2
-  we made some images to be flipped horizontally 
- fill mode we assigned it to reflect
We applied this process to the minority classes chickenpox and Measles, where for each image in these two categories 3 augmented images were made and we assigned the maximum number for each category of images was 270 image per class
While Monkeypox and Normal were not augmented
before the augmentation we had 770 image after the augmentation we have 1114 image 

# 3.Data Preprocessing
 we made label encoding where: 
 - 0 referred to normal
 - 1 referred toMonkeypox
 - 2 for Measles 
 - 3 for chickenpox
 Several image processing techniques were used to enhance the images. Sharpening was applied first but later discarded due to amplifying minor noises. Smoothing was avoided to retain important features. Adaptive histogram equalization (AHE) was applied only to the red channel in colored images to enhance the characteristic color, as applying AHE to all three channels amplified some noises. The next three images show the differences.

# 4. Feature Extraction
Two types of feature extraction are used on images after applying adaptive histogram
equalization to determine which is better after using classification model
### Manual Feature Extraction
It involves a combination of GLCM, Colour Moment, and Local Binary Pattern:

- GLCM: Gray Level Co-occurrence Matrix represents features of texture by converting a picture from RGB to shades of gray. It is aimed at measuring a matrix, with the help of pixel distance, angle of incidence, number of grey levels, and normalization for extraction of features like *contrast, dissimilarity, homogeneity, energy, and correlation*.

- Color Moment: Computes the color moments - mean, variance, skewness of each RGB channel. It changes an image to `float32` type and for every channel computes the mean, variance, and skewness then it combines them into one feature vector.

- LBP: Extracts texture features from an image by first converting it to grayscale. The LBP is then calculated based on the specified number of pixels, neighbors, radius, and method (uniform). A histogram is calculated for the LBP and normalized such that the sum of the bins equals 1.

### Pytorch Feature Extractor
using Img2Vec library which utilizes pre-trained deep learning models.it
iterates over each image to check if element is numpy array, then convert the
numpy array to PIL image to be able to extract feature vector from image.

# 5.Data Splitting Using K Fold
Evaluate a model by splitting the dataset into k equally sized folds.
Train the model on k−1folds.
Test the model on the remaining fold.
Repeat this process k times, with a different fold as the test set each time.
Average the results across all folds for a robust evaluation.

# 6.Model Comparison
we used two models for comparison 
- Random forest classifier
- XG Boost (eXtreme Gradient Boosting)

## Random Forest classifier
a supervised machine learning algorithm that builds multiple decision trees during
training and combines their outputs to improve the overall performance
#### *Random Forest Classifier with Cross Validation with the 24 manualy extracted features using the LBP, Color Moment and GLCM Result*
The model achieves an overall
accuracy of 83%.
Class 0 & Class 1 shows the
best performance with the
highest precision(0.86).
Class 3 has relatively lower
precision(0.83),
Class 2 has precision (0.79)
indicating it is less likely to
correctly identify all
instances of this class.

#### *Random Forest Classifier with Cross Validation with Pytorch features*
The model achieves an
accuracy of 90%, which is
an improvement compared
to the previous example.
Class 0 has the highest
precision (0.93), indicating
that predictions for this class
are the most reliable.
Class 1 has precision (0.91).
Class 3 has slightly lower
precision (0.89).
Class 2 has precision (0.88)

## XG Boost
a powerful and fast supervised machine
learning algorithm often used for tasks like classification, regression, and ranking. It is
based on the concept of boosting, which combines the predictions of multiple
models (typically decision trees) to create a stronger, more accurate model

#### *XG Boost with Cross Validation with the 24 manualy extracted features using the LBP, Color Moment and GLCM Result*
the model achieves overall accuracy 83%
The model performs well on Class 1 and Class 0.
Improvements are needed for Class 2 and Class 3, especially in precision and recall, to reduce false positives and false negatives.

#### *XG Boost with Cross Validation with Pytorch features*
the model achieves overall accuracy 91%
, the model performs well on class 0, 1 and 3 while it kind of struggeling with class 2. 











