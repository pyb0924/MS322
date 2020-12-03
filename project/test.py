import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == '__main__':
    print(os.getcwd())
    path = 'raw_data/instrument_dataset_2/'
    file_name = 'ground_truth/Left_Prograsp_Forceps_labels/frame000.png'
    mypath = 'img.png'

    img = cv2.imread(path + file_name, 0)
    if img is not None:
        print(np.histogram(img))
        plt.imshow(img, cmap='gray')
        plt.show()
    else:

        raise FileNotFoundError
