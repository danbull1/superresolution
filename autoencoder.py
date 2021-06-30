import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers


#from tensorflow.keras.models import load_model
weights = 'C:/DeepLearning/COMPX594/Data/autoencoder/autoencode3B'

#new_model = load_model(weights)
#print(new_model.summary())
##layer = new_model.get_layer('block1_conv1').output
#new_model.summary()
128
input_img = keras.Input(shape=(96, 96, 1))

x = layers.Conv2D(32, (2, 2), activation='relu', padding='same', name="block1_conv1")(input_img)
x = layers.Conv2D(32, (2, 2), activation='relu', padding='same', name="block1_conv2")(x)
x = layers.MaxPooling2D((2, 2), padding='same', name="block1_pool")(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="block2_conv1")(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="block2_conv2")(x)
x = layers.MaxPooling2D((2, 2), padding='same', name="block2_pool")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block3_conv1")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block3_conv2")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block3_conv3")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block3_conv4")(x)
x = layers.MaxPooling2D((2, 2), padding='same', name="block3_pool")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block4_conv1")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block4_conv2")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block4_conv3")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block4_conv4")(x)
x = layers.MaxPooling2D((2, 2), padding='same', name="block4_pool")(x)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block5_conv1")(x)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block5_conv2")(x)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block5_conv3")(x)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block5_conv4")(x)
#encoded = layers.MaxPooling2D((1, 1), padding='same', name="block5_pool")(x)

dense = layers.Dense(1000,activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3), bias_regularizer=regularizers.l2(1e-3), activity_regularizer=regularizers.l2(1e-3))(x)


#x = layers.UpSampling2D((1, 1), name="block6_unpool")(dense)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block6_deconv1")(x)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block6_deconv2")(x)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block6_deconv3")(x)
#x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block6_deconv4")(x)
x = layers.UpSampling2D((2, 2), name="block7_unpool")(dense)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block7_deconv1")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block7_deconv2")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block7_deconv3")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block7_deconv4")(x)
x = layers.UpSampling2D((2, 2), name="block8_unpool")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block8_deconv1")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block8_deconv2")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block8_deconv3")(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block8_deconv4")(x)
x = layers.UpSampling2D((2, 2), name="block9_unpool")(x)
x = layers.Conv2D(32, (3, 3), activation='relu',padding='same', name="block9_deconv1")(x)
x = layers.Conv2D(32, (3, 3), activation='relu',padding='same', name="block9_deconv2")(x)
x = layers.UpSampling2D((2, 2), name="block10_unpool")(x)
x = layers.Conv2D(32, (2, 2), activation='relu',padding='same', name="block10_deconv1")(x)
decoded = layers.Conv2D(1, (2, 2), activation='sigmoid', padding='same', name="decode")(x)


autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

import numpy as np

trainimages_npy = r'C:\DeepLearning\COMPX594\Data\autoencoder\hrpatches96.npy'
x_train = np.expand_dims(np.load(trainimages_npy,allow_pickle=True) / 255,3)
x_test = np.expand_dims(np.load(trainimages_npy,allow_pickle=True) / 255,3)

autoencoder.fit(x=x_train, y=x_train, epochs=4, batch_size=100, shuffle=True, validation_split=0.1)
autoencoder.save("autoencode5")
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(96, 96))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(96, 96))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

