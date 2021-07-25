import numpy as np
import tensorflow as tf

class_dict = {0:'dog', '1':'horse', 2:"elephant",3:"butterfly",4:"chicken",5:"cat",6:"cow",7:"sheep",8:"spider",9:"squirrel"}

# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#model = tf.keras.models.load_model("model_animal10_00.h5")
model = tf.keras.models.load_model("lenuts_00.h5")

batch_size = 5
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        directory='C:\\Data\\ANIMALS10\\split2\\test',
        target_size=(128, 128), #(224,224)
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

probabilities = model.predict_generator(test_generator, 20)
labels = test_generator.class_indices

print(probabilities)
print(np.argmax(probabilities,axis=1))