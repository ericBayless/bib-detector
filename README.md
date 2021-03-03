# Contents
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [File Directory](#File-Directory)
- [Data Dictionary](#Data-Dictionary)
- [Conclusions](#Conclusions)
- [Resources](#Resources)

# Problem Statement
In high school cross country, [chip timing](https://en.wikipedia.org/wiki/Transponder_timing) is the gold standard for accurate results.  These systems are expensive, and often beyond the budget for small race organizations and schools.  Timing can be accomplished in a relatively accurate and efficient maner using [handheld stopwatch/printers](https://www.everythingtrackandfield.com/Ultrak-L10-Lane-Timer).  However, compiling the order of finish is often a manual task prone to error.  The goal of this project is to use machine learning to build a model that can identify a racer's bib number in the [chute](https://co.milesplit.com/articles/187094/the-definitive-lexicon-of-cross-country-terms) after the finish line for the purpose of logging the order of finish.  The model needs to be lightweight and relatively fast, so that it could be used in a mobile app on a live video feed.


# Executive Summary
The first step in creating a model was to aquire image data for training and validation.  Many race directors and photography companies were contacted, but no large set of images was obtained.  Instead a small set of images from an early study in bib number detection was used called RBNR.  The dataset is comprised of 217 images containing 290 annotated racing bibs split into three sets.


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
|       |__ num_reader 
|           |__ SVHN3_custom-yolov4-tiny-detector.cfg
|           |__ SVHN3_custom-yolov4-tiny-detector_best.weights
|           |__ obj.names
|   |__ bib_detector_demo_edit.mp4
|__ Presentation
|   |__ Images
|       |__ Bib_detection_training_validation_orig.png
|       |__ Bib_detection_training_validation.png
|       |__ SVHN_training_validation.png
|   |__ bib_detector.pdf
|__ Scratch
|   |__ scratch work files
|__ Detector.py
|__ packages.txt
|__ requirements.txt
|__ README.md
|__ streamlit_app.py
```

# Data Dictionary
#### Race Bib Number (RBNR) Dataset
The dataset can be found [here](https://people.csail.mit.edu/talidekel/RBNR.html)

This image data is split into 3 sets.  Sets 1 and 2 were used to train a YOLOv4-tiny model to detect racing bibs, and set3 was used as validation for the full 2-step detector. 

#### Street View House Number (SVHN) Dataset
The dataset can be found [here]()




# Conclusions



# Resources

#### Studies
- [Racing Bib Number Recognition](https://people.csail.mit.edu/talidekel/RBNR.html)
- [Racing Bib Number Recognition Using Deep Learning](https://www.researchgate.net/publication/335234017_Racing_Bib_Number_Recognition_Using_Deep_Learning)

#### Training YOLO
- [Train YOLOv4-tiny on Custom Data - Lightning Fast Object Detection](https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/)

#### YOLO and OpenCV
- [OpenCV tutorial: YOLO - object detection](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)