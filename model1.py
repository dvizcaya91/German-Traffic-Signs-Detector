import glob

import numpy as np
from sklearn import linear_model
from PIL import Image
import matplotlib.pyplot as plt
import pickle

def ReadImages(path, get_labels=True):
	'''
	This functions read all images from a folder (using PIL library) and return them as a numpy array.
	All images are resized to 32x32 pixels, in accordance with the LeNet 5 structure.

	@path: Path where the images are store
	@get_labels: Ween set to True, the function will try to get the label from the image name (the digits after the last _ chatacter). 

	return a numpy array for the images and another one for the labels (in case it is necessary)
	'''
	print("Reading images...")
    
	#Complete the path in case is incomplete
	if path[-1] != '/':
		path += '/'

	#Read images from folder        
	images_paths = glob.glob(path+'*.ppm')
	print(len(images_paths), "images found on folder")

	#Initialize lists    
	images = []
	labels = []

	#Loop around all images on the folder    
	cont = 0
	for image_path in images_paths:
		#Open image with PIL        
		image = Image.open(image_path)
		#Resize Image        
		image = image.resize((32,32))
		#Convert image to array and normilize.
		#Append the image to the images list        
		images.append(np.array(image.getdata()).reshape(-1)/255)

		#If get_labels is set to True, get the image label        
		if get_labels:
			pos1 = image_path.rfind('_')+1 #Find the position of the last _ character
			pos2 = image_path.find('.', pos1) #Find the position of the . character
			labels.append(int(image_path[pos1:pos2])) #Get the label and append it to the labels list

		cont += 1 #Increase counter
		#If the counter is a multiple of 1000, print        
		if cont%1000 == 0:
			print(cont, "/", len(images_paths))	

	#Convert images list to numpy array            
	images = np.asarray(images)
	labels = np.asarray(labels)

	#If get_labels is set to True, return images and labels, otherwise returns just images    
	if get_labels:
		return images, labels
	else:
		return images	

def model1(c_type, image_path):
	'''
	This model identify what operation is goind to be perform with the model3 (train, test or infer) and call the a function in accordance.
	@c_type: Operation type. It could be train, test or infer
	@image_path: Path to the images folder
	'''

	model_path = 'models/model1/saved/logistic_regression_sklearn.pkl' #Path where the model is saved

	if c_type == 'train':
		#Call ReadImages function
		X, Y = ReadImages(image_path)

		print("Training...")

		#Create an object of type SGDClassifier.
		#The loss attribute is set to 'log' in order to get a logistic regression
		#The attribute shuffle is set to True in order to shuffle the images
		#The attribute verbose is set to 1 in order to for the model to output the progress.
		logistic = linear_model.SGDClassifier(loss='log', shuffle=True, verbose=1)
		
		#Call the trianing function
		logistic.fit(X, Y)
		print("Done.")

		#Print the accuracy
		print("Accuracy:", logistic.score(X, Y))

		#Save model as a pkl
		pickle.dump(logistic, open(model_path, 'wb'))

	elif c_type == 'test':	
		#Call ReadImages function
		Xtest, Ytest = ReadImages(image_path)

		#Load the regression model
		file = open(model_path,'rb')
		logistic = pickle.load(file)

		#Print the accuracy
		print(logistic.score(Xtest, Ytest))

	elif c_type == 'infer':	
		
		#Load the regression model
		file = open(model_path,'rb')
		logistic = pickle.load(file)

		#Call ReadImages function
		images_paths = ReadImages(image_path, False)
		for i in range(0, len(images_paths), 9):

			subplot_images = images_paths[i:i+9]
			index = 0
			for image in subplot_images:
				index += 1
				plt.subplot(3,3,index)
				result = logistic.predict([image])
				plt.title("Class "+str(result[0]))
				plt.axis('off')
				plt.imshow(image.reshape(32,32,-1))

			plt.show()

			inputted = input("Continue? (y/n)")
			if inputted.lower() != 'y':
				break



	