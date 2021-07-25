import numpy as np
import cv2 as cv

# img = cv.imread("IMG_0342.jpg")
# img = cv.resize(img, (1024, 1024))
#
# offx = 50
# offy = 50
# trans = np.float32([[1,0,offx],[0,1,offy]])
#
# timg = cv.warpAffine(img, trans, (1024,1024))
#
#
# cv.imshow("img", img)
# cv.imshow("trans_image", timg)
# cv.waitKey()
#

# test1 = cv.KeyPoint(0,0,0)
# test2 = cv.KeyPoint(0,1,0)
#
# print(test1)
# print(test2)
#
# pt1 = test1.pt
# pt2 = test2.pt
#
# norm = np.linalg.norm(np.subtract(pt1,pt2))
# print(norm)


class Item(object):

    def __init__(self, id):
        self.idt = id

listItems = []

listItems.append(Item(4))
listItems.append(Item(1))
listItems.append(Item(6))
listItems.append(Item(66))
listItems.append(Item(2))
listItems.append(Item(15))
listItems.append(Item(3))

# listItems = [item.idt for item in listItems]
# print(listItems)

for item in listItems:
    print(item.idt)

# To sort the list in place...
listItems.sort(key=lambda x: x.idt)

for item in listItems:
    print(item.idt)

# listItems.sort(key=id)


# newList = []
# for index,item in enumerate(listItems):
#     if(index in [1,6,7])