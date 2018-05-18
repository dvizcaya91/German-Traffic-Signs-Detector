# MODEL 2

Logistic regression using tensorflow library

## Logistic Regression Function

The model was created following a Object-Oriented approach, so a class named ANN was created
```
class ANN(object):
	"""
	This class represents a Neural Network. It will be used as a logistic regression but only adding one neuron.
	It allows to initalize the structure of the network as well as predict and train
	Methods:
		__init__: Initialize the structure of the logistic regression
		fit: Perform the training process over the network
		predict: Predict the classes of images
		score: Calculate the accuracy over a test of images
		forward: Perform the forward propagation stage of the network
	"""
```
In order to create a logistic regression, a neural network with only one neuron was defined.

```
#Initilize the weight and bias variables for the only neuron
self.W = tf.Variable((np.random.randn(D, K) * np.sqrt(2.0 / D)).astype(np.float32))
self.b = tf.Variable((np.zeros(K)).astype(np.float32))

#The logit are defined as the forward steo of one neuron.
#It doesn't have the softmax function as it is included on the cost function.
self.logits = tf.matmul(self.inputs, self.W)+self.b
```

The cost function used was the cross-entropy
```
self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
```

After testing different optimizers, the one selected was the RMS as it converged faster
```
train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(self.cost)
```

A max number of iterations of 5000 was defined. The final accuracy of the model was 97.8% over the test dataset.

## Saving Model

In order to save a load the model, the tensorflow saver was called every 20 iterations.
```
saver.save(session, 'models/model2/saved/model2')
```

## Results

The final training process obtain an accuracy of 97.8% on the test dataset.

## Usage

The script has three main functions. 

### Train

In order to train the model, perform the following command

```
python app.py train -m model2 -d images/train/
```

-d attribute indicates the directory where the images are saved

## Test

In order to test the model, perform the following command

```
python app.py test -m model2 -d images/train/
```

## Infer

In order to infer the model, perform the following command

```
python app.py infer -m model2 -d images/train/
```

This command will load a Figure for each of the images on the path. Between each images, the script will ask for confirmation to continue.

## Authors

* **Daniel Vizcaya** - *dvizcaya91@gmail.com* (https://github.com/dvizcaya91)



