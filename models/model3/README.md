# MODEL 3

Convolutional Neural Network using tensorflow library

## Convolutional Neural Network

### CNN Class
The model was created following a Object-Oriented approach, so a class named CNN was created
```
class CNN(object):
	"""
	This class represents a Convolutional Neural Network
	It allows to initalize the structure of the network as well as predict and train
	Methods:
		__init__: Initialize the structure of the network
		fit: Perform the training process over the network
		predict: Predict the classes of images
		score: Calculate the accuracy over a test of images
		forward: Perform the forward propagation stage of the network
	"""
```

One class was created for each of the layers types (convolutional, fully connected, output) also.

### Convolutional layer class
```
class ConvLayer(object):
	'''
	Convolutional Layer class
	Methods:
		__init__: Initialize the layer
		forward: Perform a forward step of the layer
	'''
	def __init__(self, cnn_filter, pooling, activation):
		'''
		Initialize the weights and bias of the layer
		@cnn_filter: List of 4 elements. The first two elements are the neighborhood connection size. The last two elements are the size of the feature map (input and output)
		@pooling: List of 4 elements. This are the neighborhood connection size for the sub-sampling layer.
		@activation: Activation function for the layer
		'''

		#Define tensorflow variables (weight and bias)
		self.W = tf.Variable((np.random.randn(*cnn_filter) * np.sqrt(2.0 / np.prod(cnn_filter[:-1]))).astype(np.float32))
		self.b = tf.Variable(np.zeros(cnn_filter[3]).astype(np.float32))

		#Define the pooling and activation as properties of the element
		self.pooling = pooling
		self.f = activation

	def forward(self, X):
		'''
		Perform a forward step of the layer
		@X: Numpy a array the is the input of the layer

		returns the output of the layer
		'''
		
		#Convolutional part
		conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
		#Add the bias
		conv_out = tf.nn.bias_add(conv_out, self.b)

		#Sub-sampling (Pooling) part
		pool_out = tf.nn.max_pool(conv_out, ksize=self.pooling, strides=[1, 2, 2, 1], padding='SAME')
		
		#Return the output using the activation function specified
		return self.f(pool_out)
```

### Fully connected layer class
```
class FullyConnectedLayer(object):
	'''
	Fully connected Layer class
	Methods:
		__init__: Initialize the layer
		forward: Perform a forward step of the layer
	'''
	def __init__(self, M1, M2, activation):
		'''
		Initialize the weights and bias of the layer
		@M1: Number of input nodes.
		@M2: Number of output nodes.
		@activation: Activation function for the layer.
		'''

		#Define tensorflow variables (weight and bias)
		self.M1 = M1
		self.M2 = M2
		W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
		b = np.zeros(M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		#Define activation as a property of the element
		self.f = activation

	def forward(self, X):
		'''
		Perform a forward step of the layer
		@X: Numpy a array the is the input of the layer

		returns the output of the layer
		'''
		#Return f((X·W+b))
		return self.f(tf.matmul(X, self.W) + self.b)
```

### Output layer
```
class OutputLayer(object):
	'''
	Output Layer class
	Methods:
		__init__: Initialize the layer
		forward: Perform a forward step of the layer
	'''
	def __init__(self, M1, M2):
		'''
		Initialize the weights and bias of the layer
		@M1: Number of input nodes.
		@M2: Number of output classes.
		'''
		#Define tensorflow variables (weight and bias)
		self.M1 = M1
		self.M2 = M2
		W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
		b = np.zeros(M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

	def forward(self, X):
		'''
		Perform a forward step of the layer without any activation function
		@X: Numpy a array the is the input of the layer

		returns the output of the layer
		'''
		#Return (X·W+b)
		return tf.matmul(X, self.W) + self.b
```

### LeNet-5 structure
The structure of the neural network was based on the LeNet-5 configuration but taking into account the colors channels of the images.

The Sub sampling layers were assumed as max pooling layers, so they are included on the ConvLayer class.
```
#1st layer (C1 and S2)
in_f = C #Input features
out_f = 6 #Output features
conv_layer = ConvLayer([5,5,in_f, out_f], pooling=[1, 2, 2, 1], activation=tf.nn.relu)

#2nd layer (C3 and S4)
in_f = out_f #Input features
out_f = 16 #Output features
conv_layer = ConvLayer([5,5,in_f, out_f], pooling=[1, 2, 2, 1], activation=tf.nn.relu) 

#3rd layer (C5)
in_f = out_f #Input features
out_f = 120 #Output features
conv_layer = ConvLayer([5,5,in_f, out_f], pooling=[1, 2, 2, 1], activation=tf.nn.relu) 

#4th layer (F6)
M1 = 120*4*4 #Input features
M2 = 84 #Output features
layer = FullyConnectedLayer(M1, M2, activation=tf.nn.tanh) 

#5th layer (Output)
M1 = M2 #Input features
M2 = K #Output features
layer = OutputLayer(M1, M2) #initialize a OutputLayer element
```

The forward function is in charge of performing the forward propagation algorithm over each layer.
```
def forward(self, X):
	'''
	This function performs the forward propagation algorithm over a set of features.
	@X: Numpy array (NxWxHxC) with th set of features

	return a numpy array of Nxn_classes
	'''
	Z = X

	#Loop around all convolutional layers
	for layer in self.conv_layers:
		#Call the forward method for every layer
		Z = layer.forward(Z)

	#Flatten the result from the last convolutional layer so it can be input on the first fully connected layer	
	Z = tf.contrib.layers.flatten(Z)
	
	#Loop around all fully connected layers	and teh output layer	
	for layer in self.layers:
		#Call the forward method for every layer
		Z = layer.forward(Z)

	return Z
```

The cost function used was the cross-entropy
```
self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
```

After testing different optimizers, the one selected was the RMS as it converged faster
```
train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(self.cost)
```

A max number of iterations of 10000 was defined. The final accuracy of the model was 99.7% over the test dataset.

## Saving Model

In order to save a load the model, the tensorflow saver was called every 20 iterations.
```
saver.save(session, 'models/model3/saved/model3')
```

## Results

The final training process obtain an accuracy of 99.7% on the test dataset.

## Usage

The script has three main functions. 

### Train

In order to train the model, perform the following command

```
python app.py train -m model3 -d images/train/
```

-d attribute indicates the directory where the images are saved

## Test

In order to test the model, perform the following command

```
python app.py test -m model3 -d images/train/
```

## Infer

In order to infer the model, perform the following command

```
python app.py infer -m model3 -d images/train/
```

This command will load a Figure for each of the images on the path. Between each images, the script will ask for confirmation to continue.

## Authors

* **Daniel Vizcaya** - *dvizcaya91@gmail.com* (https://github.com/dvizcaya91)



