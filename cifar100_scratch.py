import numpy as np
import cv2 as cv

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


meta_data_path = 'C:\\Data\\CIFAR100\\meta'
train_data_path = 'C:\\Data\\CIFAR100\\train'
test_data_path = 'C:\\Data\\CIFAR100\\test'
meta_data_dict = unpickle(meta_data_path)
train_data_dict = unpickle(train_data_path)
test_data_dict = unpickle(test_data_path)

data_train = train_data_dict[b'data']
data_test = test_data_dict[b'data']
label_train = np.array(train_data_dict[b'fine_labels'])
label_test = np.array(test_data_dict[b'fine_labels'])

# Reshape to an array of 32x32 RGB images
data_train = data_train.reshape(len(data_train), 3,32,32).transpose(0,2,3,1)
data_test = data_test.reshape(len(data_test), 3,32,32).transpose(0,2,3,1)


# index = 0
# num = 0
# while(num < 10):
#     image = data_train[index]
#     label = label_train[index]
#
#     image = image.reshape(3, 32, 32)
#     image = image.transpose(1,2,0)
#
#     # cv.imshow(str(index), image)
#     if label == 80:
#         cv.imwrite("C:\\Data\\delme\\img_" + str(index) + ".jpg", image)
#         num = num + 1
#
#     index = index + 1
