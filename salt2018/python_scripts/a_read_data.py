import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def read_all_images(path=None):
    """ Reads all of the png image in the given path.

    Parameters
    ----------
    path : string
        A path that specify where the images are located.

    Returns
    -------
    List
        A list of full path of the given images.

    """

    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    return lstFilesDCM

# define path and read images
train_image_path = "../DataSet/train/images/"
train_masks_path = "../DataSet/train/masks/"
train_image = read_all_images(train_image_path)
train_masks = read_all_images(train_masks_path)
print(len(train_image))
print(len(train_masks))

train_images_100 = []; train_masks_100 = []
for xx in range(100):
# for xx in range(len(train_masks)):
        train_images_100.append(cv2.imread(train_image[xx],0) )
        # train_images_100.append(cv2.imread(train_image[xx]) )
        train_masks_100.append(cv2.imread(train_masks[xx],0) )
train_images_100 = np.asarray(train_images_100)
train_masks_100  = np.asarray(train_masks_100)

train_images_100 = train_images_100/255.0
train_masks_100 = train_masks_100/255.0

print(train_images_100.shape)
print(train_images_100.min(),train_images_100.max())
print(train_images_100.mean(),train_images_100.std())
print(train_images_100[0].min(),train_images_100[0].max())
print(train_images_100[0].mean(),train_images_100[0].std())

print(train_masks_100.shape)
print(train_masks_100.min(),train_masks_100.max())
print(train_masks_100.mean(),train_masks_100.std())
print(train_masks_100[0].min(),train_masks_100[0].max())
print(train_masks_100[0].mean(),train_masks_100[0].std())

np.save('train_image_gray_100.npy',train_images_100)
np.save('train_masks_gray_100.npy',train_masks_100)

test_data = "../DataSet/test/images/"
test_image = read_all_images(test_data)
print(len(test_image))

test_images = []
for xx in range(len(test_image)):
        test_images.append(cv2.imread(test_image[xx],0) )
test_images = np.asarray(test_images)

test_images = test_images/255.0

print(test_images.shape)
print(test_images.min(),test_images.max())
print(test_images.mean(),test_images.std())
print(test_images[0].min(),test_images[0].max())
print(test_images[0].mean(),test_images[0].std())

np.save('test_image_gray_100.npy',test_images)


# =========================================
# Def: Show some of the images
# for iter in range(len(train_masks_100)):
#
#     plt.subplot(2,1,1)
#     plt.axis('off')
#     plt.imshow(train_images_100[iter],cmap='gray')
#
#     plt.subplot(2,1,2)
#     plt.imshow(train_masks_100[iter],cmap='gray')
#     plt.axis('off')
#     plt.title(train_masks_100[iter].sum())
#     plt.show()
# =========================================













# -- end code --
