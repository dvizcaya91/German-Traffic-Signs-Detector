import glob
from tqdm import tqdm
import requests, zipfile, io, math
from PIL import Image
import numpy as np
from zipfile import ZipFile

def Download():
	'''
	This function download the zip, saves it and extract all
	'''
	zip_file_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
	print("Downloading file")

	#Get request to the URL
	r = requests.get(zip_file_url, stream=True) 

	#Script to show the progress bar
	total_size = int(r.headers.get('content-length', 0)); #Total size of file
	block_size = 1024 #Size of a block
	wrote = 0 

	with open('output.zip', 'wb') as f: #Writing file
	    for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
	        wrote = wrote  + len(data)
	        f.write(data)
	if total_size != 0 and wrote != total_size:
	    print("ERROR, something went wrong")  
	
	#Read the zip file    
	data = ZipFile('output.zip')  
	#Extract all
	data.extractall()  

	#Call the function to process images
	SaveImages()

def SaveImages():
	'''
	This function read the images that were downloaded and save them on two different folder: Train and test
	'''

	#Initialize lists
	totals = []
	images_files = []

	#Loop around the folders of each class
	for c in range(0, 43):

		#Define folder name
		if c < 10:
			files_path = 'GTSRB/Final_Training/Images/0000'+str(c)
		else:
			files_path = 'GTSRB/Final_Training/Images/000'+str(c)	

		c_images_files = glob.glob(files_path+'/*.ppm') #Get all images on the directory
		images_files.append(c_images_files) #Append list to the images_files list
		totals.append(len(images_files[c])) #Append the number of images to the totals list

	#Get the number of images of the class with more examples	
	totals = np.asarray(totals)
	max_total = np.max(totals)

	c = 0	
	#Loop around each class
	for files in images_files:
		cont = 0

		#Number of examples on the class that are going to be part of the training dataset
		class_size = len(files)*0.8

		#Relation between the class with more examples and the number of examples of current class
		repetitions = int(max_total/totals[c])

		#Loop around the images of the class
		for image_file in files:

			cont += 1
			#open the image	
			image = Image.open(image_file)

			#Define if it is a train or test image
			if cont < class_size:
				folder_name = 'images/train/'
			else:
				folder_name = 'images/test/'

			#Save the image as many times as need in order to balance the classes	
			for i in range(0, repetitions):
				image_name = str(cont)+'_'+str(i)+'_'+str(c)		
				filename = folder_name+image_name	
				image.save(filename + '.ppm')

		print(c, "class completed", cont)	

		c += 1
