
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#Initiallising the CNN
Classifier = Sequential()

#Step-1 Convolution
Classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation="relu"))

#Step-2 - Pooling
Classifier.add(MaxPool2D(pool_size=(2,2)))

#Adding Second layer
Classifier.add(Conv2D(32,(3,3), activation='relu'))

Classifier.add(MaxPool2D(pool_size=(2,2)))

#Step-3 Flattening
Classifier.add(Flatten())

#Step-4 Full Connection
Classifier.add(Dense(units=128,activation='relu'))

Classifier.add(Dense(units=1,activation="sigmoid"))

#Compiling the CNN
Classifier.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

#PArt-2 Flitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip = True )

test_data = ImageDataGenerator(rescale = 1./255)

training_set = train_data.flow_from_directory(r'E:\Deep learning\CNN\Images',target_size = (64,64),batch_size = 32,class_mode='binary')

test_set = test_data.flow_from_directory(r'E:\Deep learning\CNN\Images',target_size = (64,64),batch_size=32,class_mode='binary')

model =Classifier.fit_generator(training_set,
                                steps_per_epoch=80,
                                epochs=1,
                                validation_data = test_set,
                                validation_steps = 200)

Classifier.save("model.h5")
print("Saved model to disk")
