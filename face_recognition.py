from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from PIL import Image  
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import argparse
import numpy as np
import requests 
from sklearn.svm import LinearSVC

# take arguments from terminal
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
	help="search query to search Bing Image API for")
args = vars(ap.parse_args())

file_path = args["query"]

print(file_path)
def markAttendance(data):
	print("data in mark markAttendance ===>", data)
	URL = "http://localhost:4000/attendance/fill-attendance"
	data = {'userId':"5d941e75115e4b3dbcf8c4b9"}
	r = requests.post(url = URL, data = data)
	pastebin_url = r.json()
	print("response of API ==========>", pastebin_url)  	



def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(path):
	faces = list()
	# get face
	face = extract_face(path)
	# store
	faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(file_path):
	print("directory =======>", file_path)
	X= list()
	print("x , y  @54=======>", X)
	# enumerate folders, on per class
	faces = load_faces(file_path)
	# create labels
	# summarize progress
	# print('>loaded %d examples for class: %s' % (len(faces), subdir))
	# store
	X.extend(faces)
		
		
	return asarray(X)



testX = load_dataset(file_path)
# print("testX1 ================>", testX)
# testy = ['ranveer-singh','shahid-kapoor','raam','pushpraj','akshay-kumar']
testy = ['amir-khan', 'sanjay-dutt' ,'ranveer-singh', 'john-cena', 'vin deisel', 'ben_afflek', 'robert downey jr', 'jerry_seinfeld', 'amitab bachan', 'shahid-kapoor', 'kuldip-shiddhpura', 'jonny depp', 'raam', 'elton_john', 'madonna', 'mindy_kaling', 'pushpraj', 'ranveer-kapoor', 'akshay-kumar', 'arijit singh']




# testy = ['shahrukh-khan','raam']


# print("test X ======================>", testX, range(testX.shape[0]))



# load faces
data = load('dataset.npz')
# testX_faces =  data['arr_2']
testX_faces = testX
# load face embeddings
data = load('dataset.npz')
# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# print("testX_faces ========> ", testX_faces[2])
trainX, trainy = data['arr_0'], data['arr_1']

# print("testX ===============> ", testX[3])
# print("trainy {} : texty {} : ", trainy, testy)
# # normalize input vectors
in_encoder = Normalizer(norm='l2')

# Convert from 4D to 2D array.
dataset_size = len(trainX)
TwoDim_dataset_trainX = trainX.reshape(dataset_size,-2)
# label encode targets
trainX = in_encoder.transform(TwoDim_dataset_trainX)

# Convert from 4D to 2D array.
# dataset_size = len(testX)
# TwoDim_dataset_testX = testX.reshape(dataset_size,-2)
# testX = in_encoder.transform(TwoDim_dataset_testX)

dataset_size = len(testX)
TwoDim_dataset_testX = testX.reshape(dataset_size,-2)
testX = in_encoder.transform(TwoDim_dataset_testX)



print("outcoder")
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# print(out_encoder.transform(data['arr_3']))
# fit model
# model = SVC(kernel='linear', probability=True)
clf = LinearSVC(random_state=0, tol=1e-5)
print("model fit")
# model.fit(trainX, trainy,  sample_weight=None)
clf.fit(trainX, trainy.ravel()) 
print("Model fit done")
# test model on a random example from the test dataset
# print("range(testX.shape[0]) ==========>", range(testX.shape[0]))
# selection = choice([i  for i in range(testX.shape[0])])
# print("selection ==========>", selection)

print("Seelctoin Done") 



# random_face_pixels = testX_faces[selection]
# random_face_emb = testX[selection]
# print(" =============> ", testX[selection] == testX1)
# random_face_class = testy[selection]

random_face_pixels = testX_faces[0]
random_face_emb = testX[0]
# random_face_class = testy[2]


# random_face_name = out_encoder.inverse_transform([random_face_class])
# print(" random_face_name ==============>", random_face_name)
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
dataset_size = len(samples)
samples = samples.reshape(dataset_size,-1)
print("before predict");
yhat_class = clf.predict(samples)
print("yhat_class ===========> ", yhat_class)
yhat_prob = clf.score(samples)
# get name
class_index = yhat_class[0]
# class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print("PRINT NAMES ============>", predict_names[0], yhat_prob)
# print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
# print('Predicted: %s (%.3f)' % (predict_names[0]))

# API CALL IS PREDICTION GET RIGHT
# if(predict_names[0] == 'raam'):
# 	markAttendance(predict_names[0])



# print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
# title = '%s (%.3f)' % (predict_names[0], class_probability)
# title = '%s (%.3f)' % (predict_names[0])
pyplot.title(predict_names[0])
pyplot.show()





