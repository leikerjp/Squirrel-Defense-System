import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint


# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# # add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer -- we have 10 classes
# predictions = Dense(10, activation='softmax')(x)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)


# x = Flatten()(x)
# x = BatchNormalization(axis=0)(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# # x = BatchNormalization(axis=0)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# # x = BatchNormalization(axis=0)(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = BatchNormalization(axis=0)(x)

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = AveragePooling2D(pool_size=(7,7))(x)
# x = Flatten(name="flatten")(x)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(10, activation='softmax')(x)




# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all but the last convolutional ResNet50 layers
for layer in base_model.layers[:143]:
    layer.trainable = False
#
# # Check that we set the freezes correctly
# for i,layer in enumerate(base_model.layers):
#         print(i, layer.name, "-", layer.trainable)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

batch_size = 8
train_generator = train_datagen.flow_from_directory(
        directory='C:\\Data\\ANIMALS10\\split\\train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        directory='C:\\Data\\ANIMALS10\\split\\val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


check_point = ModelCheckpoint(filepath="animal10.h5",
                              monitor='val_accuracy',
                              mode="max",
                              save_best_only=True,
                              )

# train the model on the new data for a few epochs
model.fit(
        train_generator,
        #steps_per_epoch=19631 / batch_size,
        epochs=5,
        validation_data=validation_generator,
        #validation_steps=3922 / batch_size,
        callbacks=[check_point]
        )



# # at this point, the top layers are well trained and we can start fine-tuning
# # convolutional layers from inception V3. We will freeze the bottom N layers
# # and train the remaining top layers.
#
# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)
#
# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True
#
# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from tensorflow.keras.optimizers import SGD
#
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# train the model on the new data for a few epochs
# model.fit(
#         train_generator,
#         steps_per_epoch=19631 / batch_size,
#         epochs=20,
#         validation_data=validation_generator,
#         validation_steps=3922 / batch_size)

#model.save_weights('C:\\Users\\jplei\\Workspace\\Squirrel-Defense-System\\weights_01.h5')  # always save your weights after training or during training

model.save("model_animal10_00.h5")
print("Duuuuuuuuuuude we made it!")