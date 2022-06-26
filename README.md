# How_I_Am_Learning_ML

## 1. Problem Statement: Train a digit classifier.
Major part in this problem statement is model selection and approach.
As you know these are basic steps of model training.

<sub>
  Step 1: Data Gathering 
  Step 2: Data Labeling
  Step 3: Train Model 
  Step 4: Test Model
</sub>

##### Please refer folder -> "Train_Classifier", here i have trained and evaluate classifier with different models.

Learnings: 
1. I have trained digit dataset with different classifier like ->
   1. Logistic Regression 
   2. Random Forest 
   3. SVM 
   4. Deep Neural Network 
   5. Convolution Neural Network
2. Model testing has big role in training pipeline, as we are not sure 'How model will perform on realtime world'. 
   1. Cross Validation , Confusion Matrix and calculating loss helps here.

## 2. Problem Statement: Train a Face Detector.
Face Detection is not that simple as to train a digit classifier. In digit classifier eventually, 
we are giving positive and negative samples but here will only have to train on positive samples. Labeling has major 
role in Face Detection, face has to label properly for that there is open source tool LabelImage.
In case of object detection, theory says that NN outperforms all other ML architectures. 

Learnings: 
1. I have trained ->
   1. HOG+SVM - Classifier 
   2. HOG+SVM - Detector(Dlib)
2. Explored ->
   1. ImgLab - It's an annotation tool which used while labeling dataset for dlib based detector.
   2. MTCNN - (https://github.com/ipazc/mtcnn)
   3. Viola-Jones(Haar-cascade Detection in OpenCV)
 
   