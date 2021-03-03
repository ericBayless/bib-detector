# Contents
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Conclusions](#Conclusions)
- [File Directory](#File-Directory)
- [Data Description](#Data-Description)
- [Resources](#Resources)

# Problem Statement
In high school cross country, [chip timing](https://en.wikipedia.org/wiki/Transponder_timing) is the gold standard for accurate results.  These systems are expensive, and often beyond the budget for small race organizations and schools.  Timing can be accomplished in a relatively accurate and efficient maner using [handheld stopwatch/printers](https://www.everythingtrackandfield.com/Ultrak-L10-Lane-Timer).  However, compiling the order of finish is often a manual task prone to error.  The goal of this project is to use machine learning to build a model that can identify a racer's bib number in the [chute](https://co.milesplit.com/articles/187094/the-definitive-lexicon-of-cross-country-terms) after the finish line for the purpose of logging the order of finish.  The model needs to be lightweight and relatively fast, so that it could be used in a mobile app on a live video feed.


# Executive Summary
The first step in creating a model was to aquire image data for training and validation.  Many race directors and photography companies were contacted, but no large set of images was obtained.  Instead a small set of images from an early study in bib number detection was used called RBNR.  The dataset is comprised of 217 images containing 290 annotated racing bibs split into three sets.  A link to the original study and images is provided below.  

Because of the limited number of images, a strategy was devised to use part of this dataset (set 1 and set 2) for training a model to detect racing bibs, and another dataset to train a model to read numbers.  The second model could then be used to read the digits on the bibs detected by the first model.  The dataset used for the second model was the Street View House Number (SVHN) dataset.  A similar strategy was used in another study using deep learning linked in the resources below.  However, this study used a combination of racing images and SVHN images for the best result.

The next step was to build and train the models.  Computer vision is a well studied and rapidly growing field.  Many models based on well researched algorithms are available, and can be retrained for custom use cases.  Rather than reinvent the wheel, focus was given to selecting one of these models.  Based on its reputation as a fast and lightweight model that doesn't give up much performance relative to its more resource intensive competition, the Darknet YOLOv4-tiny model was chosen.  

Two custom models were created by retraining YOLOv4-tiny; one for detecting the bibs in an image, and the other for detecting the digits on those bibs.  Initially 127 of the RBNR images (set 1 and set 2) were used to train the bib detection model.  However, its detection ability was low with an average precision score of 76.03% at an IoU threshold of 50%.  This was overcome by using image augmentation on the original 127 images to create 5088 images.  This increased the average precision on the augmented images used for validation to 94.42%.  For the digit detection model, the mean average precision on the SVHN test images after training was 84.12% at an IoU threshold of 50%.

Further testing was conducted with the digit detection model by cropping out the bibs from all three sets of the RBNR dataset and comparing the results to the true values.  Even though the detector predicts individual digits, only a fully matching bib number was counted as correct.  In the expected use case for determining order of finish, a partial match is not of more value than no match.  The result of this test was an accuracy of 67.59%.  The full end to end model (bib detection + digit detection) was then tested using set 3 of the RBNR dataset.  Again only a full match was counted as correct.  The result of this test was an accuracy of 38.05%.  On further review it was observed that set 3 images contained a unique font for the number one which the model had difficulty recognizing, which partially contributed to the low score.

# Conclusions
Taken at face value, the low accuracy scores would indicate an underperforming application.  However, the validation set used is limited in number.  Also, it does not represent the planned use of the model well.  Most of the images are action shots containing many runners, whereas, in practice the model is intended to be used in a live feed on one runner at a time.  The runner could be positioned within the frame where the model is able to read the number.  Subjectively, in testing the application with a live webcame feed, it does reasonably well at detecting bibs and reading the numbers.

Given the limitations of the current data, the next steps will include gathering more images and videos specific to the desired usecase for further testing and training of the models.  Also, to better understand performance, the models will be ported to an iOS app where true performance metrics can be obtained.  It would also be desirable to test the same process with other algorithms for comparison to YOLOv4-tiny on size, speed, and accuracy.

# File Directory
```
project-3
|__ Code
|   |__ 01_Preprocessing_SVHN.ipynb   
|   |__ 02_SVHN_YOLOv4_tiny_Darknet_Roboflow.ipynb   
|   |__ 03_Digit_Detector_Validation.ipynb
|   |__ 04_Preprocessing_RBNR.ipynb
|   |__ 05_RBNR_YOLOv4_tiny_Darknet_Roboflow.ipynb
|   |__ 06_Full_Detector_Validation_Demo.ipynb
|__ Data
|   |__ YOLO
|       |__ bib_detector 
|           |__ RBNR2_custom-yolov4-tiny-detector.cfg
|           |__ RBNR2_custom-yolov4-tiny-detector_best.weights
|           |__ obj.names
|       |__ num_reader 
|           |__ SVHN3_custom-yolov4-tiny-detector.cfg
|           |__ SVHN3_custom-yolov4-tiny-detector_best.weights
|           |__ obj.names
|   |__ bib_detector_demo.mp4
|__ Presentation
|   |__ Images
|       |__ Bib_detection_training_validation_orig.png
|       |__ Bib_detection_training_validation.png
|       |__ SVHN_training_validation.png
|   |__ RBN_detector.pdf
|__ Scratch
|   |__ scratch work files
|__ Detector.py
|__ packages.txt
|__ requirements.txt
|__ README.md
|__ streamlit_app.py
```

# Data Description
#### Race Bib Number (RBNR) Dataset
The dataset can be found [here](https://people.csail.mit.edu/talidekel/RBNR.html)

This image data is split into 3 sets.  Sets 1 and 2 were used to train a YOLOv4-tiny model to detect racing bibs, and set3 was used as validation for the full 2-step detector. 

#### Street View House Number (SVHN) Dataset
The dataset can be found [here](http://ufldl.stanford.edu/housenumbers/)

This dataset contains a set of full numbers split into train, test, and extra, as well as cropped digits with the same catagorization.  Only the train and test sets of the full numbers were used.  A YOLOv4-tiny model was retrained to detect digits using the training set, and validated against the test set. 

# Resources
#### Studies
- [Racing Bib Number Recognition](https://people.csail.mit.edu/talidekel/RBNR.html)
- [Racing Bib Number Recognition Using Deep Learning](https://www.researchgate.net/publication/335234017_Racing_Bib_Number_Recognition_Using_Deep_Learning)

#### Training YOLO
- [Train YOLOv4-tiny on Custom Data - Lightning Fast Object Detection](https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/)

#### YOLO and OpenCV
- [OpenCV tutorial: YOLO - object detection](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)