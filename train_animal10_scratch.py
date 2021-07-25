import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

width = 128
height = 128
dimensions = (width,height,3)

inputs = tf.keras.Input(shape=dimensions)
x = Conv2D(filters=32,
           kernel_size=(5,5),
           padding='same',
           input_shape=dimensions,
           activation='relu',
           )(inputs)
x = MaxPool2D(strides=(2,2))(x)
x = Conv2D(filters=48,
           kernel_size=(5,5),
           padding='valid',
           activation='relu',
           )(x)
x = MaxPool2D(strides=(2,2))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(84, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions, name='lenuts')
model.summary()


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

batch_size = 64
train_generator = train_datagen.flow_from_directory(
        directory='C:\\Data\\ANIMALS10\\split\\train',
        target_size=(width,height),
        batch_size=batch_size,
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        directory='C:\\Data\\ANIMALS10\\split\\val',
        target_size=(width,height),
        batch_size=batch_size,
        class_mode='categorical')

check_point = ModelCheckpoint(filepath="lenuts_check.h5",
                              monitor='val_accuracy',
                              mode="max",
                              save_best_only=True,
                              )

# train the model on the new data for a few epochs
model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        callbacks=[check_point]
        )

model.save("lenuts_00.h5")
print("Duuuuuuuuuuude we made it!")


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, Activation, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, Flatten
# from tensorflow.keras.callbacks import ModelCheckpoint
# import os
# from sklearn.utils import class_weight
#
# # Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
# width = 128
# height = 128
# dimensions = (width,height,3)
#
# inputs = tf.keras.Input(shape=dimensions)
# x = Conv2D(filters=32,
#            kernel_size=(5,5),
#            padding='same',
#            input_shape=dimensions,
#            activation='relu',
#            )(inputs)
# x = MaxPool2D(strides=(2,2))(x)
# x = Conv2D(filters=48,
#            kernel_size=(5,5),
#            padding='valid',
#            activation='relu',
#            )(x)
# x = MaxPool2D(strides=(2,2))(x)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dense(84, activation='relu')(x)
# predictions = Dense(1, activation='softmax')(x)
#
# model = Model(inputs=inputs, outputs=predictions, name='lenuts')
# model.summary()
#
#
# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#
# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
# val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rescale=1./255)
#
# batch_size = 64
# train_generator = train_datagen.flow_from_directory(
#         directory='C:\\Data\\ANIMALS10\\split2\\train',
#         target_size=(width,height),
#         batch_size=batch_size,
#         class_mode='binary')
# validation_generator = val_datagen.flow_from_directory(
#         directory='C:\\Data\\ANIMALS10\\split2\\val',
#         target_size=(width,height),
#         batch_size=batch_size,
#         class_mode='binary')
#
# check_point = ModelCheckpoint(filepath="lenuts_check_bin.h5",
#                               monitor='val_accuracy',
#                               mode="max",
#                               save_best_only=True,
#                               )
#
# # train the model on the new data for a few epochs
# nsq = os.listdir('C:\\Data\\ANIMALS10\\split2\\train\\not_squirrel')
# sq = os.listdir('C:\\Data\\ANIMALS10\\split2\\train\\squirrel')
# nsq = len(nsq)
# sq = len(sq)
# class_weights = (nsq + sq) / (2 * np.array([nsq,sq]))
# # class_weights = class_weight.compute_class_weight('balanced',
# #                                                   np.array([len(nsq),len(sq)]),
# #                                                   np.array(['not_squirrel','squirrel']))
# model.fit(
#         train_generator,
#         epochs=3,
#         class_weight={0:class_weights[0],1:class_weights[1]},
#         validation_data=validation_generator,
#         callbacks=[check_point]
#         )
#
# model.save("lenuts_bin_00.h5")
# print("Duuuuuuuuuuude we made it!")