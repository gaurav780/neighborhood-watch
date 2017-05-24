from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from scipy.misc import imread, imresize
from keras.datasets import mnist
import pickle as pkl
import cv2
#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

img1 = imread('streetview.jpg', mode='RGB')
img1 = imresize(img1, (224, 224))
print 'img1 shape'
print img1.shape

#Create your own input format (here 3x200x200)
input1= Input(shape=(224,224,3), name='image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input1)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(3, activation='softmax', name='predictions')(x)

#Create your own model 
my_model = Model(input=input1, output=x)
my_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# loading the data
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
lat_longs =pkl.load(open("lat_longs_1000.p","rb"))
ll_to_buckets = pkl.load(open("ll_to_buckets_1000.p","rb"))
y_train = []
x_train =[]
for ll in lat_longs[:700]:
	y_train.append(ll_to_buckets[ll])
	img1 = imread('images/'+ll[0]+'_'+ll[1]+'_60.000000.png', mode='RGB')
	img1 = imresize(img1, (224, 224))
	x_train.append(img1)
y_train = np.array(y_train)
x_train = np.array(x_train)

y_test = []
x_test =[]
for ll in lat_longs[700:]:
        y_test.append(ll_to_buckets[ll])
        img1 = imread('images/'+ll[0]+'_'+ll[1]+'_60.000000.png', mode='RGB')
        img1 = imresize(img1, (224, 224))
        x_test.append(img1)
y_test = np.array(y_test)
x_test = np.array(x_test)
print 'Shapes\n'
print x_train.shape
print y_train.shape
print '\n'
'''
# converting it to RGB
x_train = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')
print 'x_train shape'
print x_train.shape
print '\n'

x_test = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')
print 'x_test shape'
print x_test.shape
print '\n'
'''

#model = my_model(x_train.shape[1:], len(set(y_train)), 'relu')
hist = my_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)    
#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()
