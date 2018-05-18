# MODEL 1

Logistic regression using scikit learn library

## Logistic Regression Function

This model used the SGDClassifier from the scikit learn library. In order to make the classifier a logistic regression, the attribute loss was set to log. 

```
sklear.linear_model.SGDClassifier(loss='log')
```

## Saving Model

In order to save a load the model, the pikle library was used.

## Results

The final training process obtain an accuracy of 86.9% on the test dataset.

## Usage

The script has three main functions. 

### Train

In order to train the model, perform the following command

```
python app.py train -m model1 -d images/train/
```

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



