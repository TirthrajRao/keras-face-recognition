# from keras.models import load_model
# # load the model
# model = load_model('facenet_keras.h5')
# # summarize input and output shape
# print(model.inputs)
# print(model.outputs)

# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import face_recognition
# extract multiple face from a given photograph
# def extract_face(filename, required_size=(160,160,)):

	# image = face_recognition.load_image_file(filename)
	# image = image.convert('RGB')
	# face_locations = face_recognition.face_locations(image)
	# print("FACE LOCATION ===========>",face_locations)
	# print(f'There are {len(face_locations)} people in this image')
	# detector = MTCNN()
	# # detect faces in the image
	# results = detector.detect_faces(face_locations[0])
	# # extract the bounding box from the first face
	# x1, y1, width, height = results[0]['box']
	# # bug fix
	# x1, y1 = abs(x1), abs(y1)
	# x2, y2 = x1 + width, y1 + height
	# # extract the face
	# face = pixels[y1:y2, x1:x2]
	# # resize pixels to the model size
	# image = Image.fromarray(face)
	# image = image.resize(required_size)
	# face_array = asarray(image)
	# image.show()

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# image.show();
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	print("pixeks as arrya =============<>", pixels)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	print(f'There are {len(results)} people in this image')
	i= 0
	for filename in results:
		i = i + 1
		# extract the bounding box from the first face
		x1, y1, width, height = filename['box']
		# bug fix
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		print("face array %d ====>",i , face_array )
		image.show()
	

# load the photo and extract the face
extract_face('./3group1.jpg');
# print("pixels ======>", pixels);