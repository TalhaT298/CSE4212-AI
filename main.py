import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.applications import vgg16
inputs = Input((224, 224, 3))
x = Conv2D(filters = 64, kernel_size=(3,3), activation = 'relu', name = 'block1_conv1', padding='same')(inputs)
x = Conv2D(filters = 64, kernel_size=(3,3), activation = 'relu',name = 'block1_conv2', padding='same')(x)
x = MaxPooling2D(name='block1_pool')(x)

x = Conv2D(filters = 128, kernel_size=(3,3), activation = 'relu', name = 'block2_conv1', padding='same')(x)
x = Conv2D(filters = 128, kernel_size=(3,3), activation = 'relu', name = 'block2_conv2', padding='same')(x)
x = MaxPooling2D(name='block2_pool')(x)

x = Conv2D(filters = 256, kernel_size=(3,3), activation = 'relu', name='block3_conv1', padding='same')(x)
x = Conv2D(filters = 256, kernel_size=(3,3), activation = 'relu', name='block3_conv2', padding='same')(x)
x = Conv2D(filters = 256, kernel_size=(3,3), activation = 'relu', name='block3_conv3', padding='same')(x)
x = MaxPooling2D(name='block3_pool')(x)

x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block4_conv1', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block4_conv2', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block4_conv3', padding='same')(x)
x = MaxPooling2D(name='block4_pool')(x)

x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv1', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv2', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv3', padding='same')(x)
x = MaxPooling2D(name='block5_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name = 'fc1')(x)
x = Dense(4096, activation='relu', name = 'fc2')(x)

outputs = Dense(1000, activation = 'sigmoid',name = 'output_layer')(x)
model = Model(inputs, outputs, name='my_vgg16_model')
vgg_model = vgg16.VGG16()

for i in range(len(model.layers)):
  customized_layer = model.layers[i]
  vgg16_layer = vgg_model.layers[i]

  if customized_layer.name == vgg16_layer.name:
    customized_layer.set_weights(vgg16_layer.get_weights())

model.summary()

#2nd
inputs = Input((321, 321, 4))
x = Conv2D(filters = 64, kernel_size=(3,3), activation = 'relu', name = 'block1_conv1', padding='same')(inputs)
x = Conv2D(filters = 64, kernel_size=(3,3), activation = 'relu',name = 'block1_conv2', padding='same')(x)
x = MaxPooling2D(name='block1_pool')(x)

x = Conv2D(filters = 128, kernel_size=(3,3), activation = 'relu', name = 'block2_conv1', padding='same')(x)
x = Conv2D(filters = 128, kernel_size=(3,3), activation = 'relu', name = 'block2_conv2', padding='same')(x)
x = Conv2D(filters = 128, kernel_size=(3,3), activation = 'relu', name = 'block2_conv3', padding='same')(x)
x = Conv2D(filters = 128, kernel_size=(3,3), activation = 'relu', name = 'block2_conv4', padding='same')(x)
x = MaxPooling2D(name='block2_pool')(x)

x = Conv2D(filters = 256, kernel_size=(3,3), activation = 'relu', name='block3_conv1', padding='same')(x)
x = Conv2D(filters = 256, kernel_size=(3,3), activation = 'relu', name='block3_conv2', padding='same')(x)
x = Conv2D(filters = 256, kernel_size=(3,3), activation = 'relu', name='block3_conv3', padding='same')(x)
x = MaxPooling2D(name='block3_pool')(x)

x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block4_conv1', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block4_conv2', padding='same')(x)
x = MaxPooling2D(name='block4_pool')(x)

x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv1', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv2', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv3', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv4', padding='same')(x)
x = Conv2D(filters = 512, kernel_size=(3,3), activation = 'relu', name='block5_conv5', padding='same')(x)
x = MaxPooling2D(name='block5_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name = 'fc1')(x)
x = Dense(4096, activation='relu', name = 'fc2')(x)
x = Dense(2048, activation='relu', name = 'fc3')(x)
x = Dense(2048, activation='relu', name = 'fc4')(x)
x = Dense(1024, activation='relu', name = 'fc5')(x)

outputs = Dense(1234, activation = 'sigmoid',name = 'output_layer')(x)
model = Model(inputs, outputs, name='my_customized_model')

model.summary()