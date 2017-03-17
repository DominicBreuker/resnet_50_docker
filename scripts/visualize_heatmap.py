import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

#############################
#
# To run the visualizations, you must have created heatmap data before
# Use the extraction mode to create an INDEX_FILE and EXTRACTIONS_FILE
# Then replace the paths below with those to your files
#
#############################

IMAGES_DIR = 'data'
INDEX_FILE = 'output/index_resnet50_300x300-FCN_True-custom_2017-02-14-07-50-32.npy'
EXTRACTIONS_FILE = 'output/extractions_resnet50_300x300-FCN_True-custom_2017-02-14-07-50-32.npy'


def scale_heatmap(img, heatmap):
    new_shape = (img.shape[0], img.shape[1])
    heatmap = imresize(heatmap, new_shape, interp='bicubic')
    top_bot = img.shape[1] - new_shape[1]
    top = top_bot // 2
    bot = top_bot - top
    lef_rig = img.shape[0] - new_shape[0]
    lef = lef_rig // 2
    rig = lef_rig - lef
    heatmap = np.lib.pad(heatmap, ((lef, rig), (top, bot)), 'constant',
                         constant_values=(0))
    heatmap[np.where(heatmap < 100)] = 0
    return heatmap


def display_image(img, heatmap=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.hold(True)
    if heatmap is not None:
        plt.imshow(heatmap, alpha=0.2)
    plt.show()

index = np.load(INDEX_FILE).astype(str)
data = np.load(EXTRACTIONS_FILE)

for i in range(len(index)):
    current_image = imread(os.path.join(IMAGES_DIR, index[i]), mode='RGB')
    current_feature_map = data[i][:, :, 0]
    current_heatmap = scale_heatmap(current_image, current_feature_map)
    print("Max: {:2.2f}, Min: {:2.2f}, Mean: {:2.2f}"
          .format(np.max(current_feature_map),
                  np.min(current_feature_map),
                  np.mean(current_feature_map)))
    display_image(current_image, current_heatmap)
