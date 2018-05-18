# Deep Learning Challenge

This repository was created by Daniel Vizcaya (dvizcaya91@gmail.com)

## Getting Started

These repository is the solution to the Kiwi Campus Deep Learning Challenge. 

### Installing required libraries

Several libraries had been used for the challenge and they are listed on requirements.txt

```
pip install -r requirements.txt
```

## Downloading images

In order to download images in the correct format, call the app.py function with the download argument

```
python app.py download
```

This will download the zip file from http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip

The script will unzip the file and save the images in the correct format 

It will create an images folder with two folders inside it (train and test). The train folder will contain 80% of the images, while the test folder will contain the other 20%.

As the dataset classes are imbalance, the script will duplicate images of the classes with lower samples, until all classes are balance.

The last number of the image name represents the class of the image. For example:

1000_0_17.ppm will be an image of class 17.

## Models

The script contains 3 models (model1.py, model2.py and model3.py). Each model has a Jupyter report under the reports folder.

### Train

In order to train the model, perform the following command

```
python app.py train -m model1 -d images/train/
```

-m attribute indicates the model to use
-d attribute indicates the directory where the images are saved

## Test

In order to test the model, perform the following command

```
python app.py test -m model1 -d images/train/
```

## Infer

In order to infer the model, perform the following command

```
python app.py infer -m model1 -d images/train/
```

This command will load a Figure for each of the images on the path. Between each images, the script will ask for confirmation to continue.

## Authors

* **Daniel Vizcaya** - *dvizcaya91@gmail.com* (https://github.com/dvizcaya91)



