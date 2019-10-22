import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import random
import math


def show_class_percentage(data_path, starting_patient, end_patient):
    # 0: background, 1: kidney, 2: tumor
    class_percentage = {0: 0, 1: 0, 2: 0}
    total_number_images = 0.0
    for i in range(starting_patient, end_patient+1):
        case_id = str(i).zfill(5)
        mask_name = ''
        for file in os.listdir(data_path):
            if str(case_id) in file and 'segmentation' in file:
                mask_name = file
                mask = nib.load(os.path.join(data_path, mask_name))
                mask = mask.get_data()
                for j in range(mask.shape[0]):
                    total_number_images += 1
                    classes = np.unique(mask[j])
                    for element in classes:
                        class_percentage[element] += 1

    print("Background class exists in {0:.2f} % of images ".format((class_percentage[0]/total_number_images)*100))
    print("Kidney class exists in {0:.2f} % of images ".format((class_percentage[1] / total_number_images)*100))
    print("Tumor class exists in {0:.2f} % of images ".format((class_percentage[2] / total_number_images)*100))

    return mask


def show_random_cases(data_path, starting_patient, end_patient, number = 10, has_tumor= True):
    output_images = []
    for i in range(number):
        case_id = random.randint(starting_patient, end_patient)
        case_id = str(case_id).zfill(5)
        img_name = ''
        mask_name = ''
        for file in os.listdir(data_path):
            if img_name != '' and mask_name != '':
                break
            if str(case_id) in file and 'imaging' in file:
                img_name = file
            elif str(case_id) in file and 'segmentation' in file:
                mask_name = file
        vol = nib.load(os.path.join(data_path, img_name))
        mask_vol = nib.load(os.path.join(data_path, mask_name))
        img = vol.get_data()
        mask = mask_vol.get_data()
        for j in range(mask.shape[0]):
            if has_tumor:
                if np.max(mask[j]) == 2:
                    output_images.append((img[j],mask[j]))
                    break
            else:
                if np.max(mask[j]) != 2:
                    output_images.append((img[j],mask[j]))
                    break

    for i in range(number):
        f, axarr = plt.subplots(1, 2, figsize=(15, 15))
        image, mask = output_images[i]
        axarr[0].imshow(image, cmap = plt.cm.gray)
        axarr[1].imshow(mask, cmap = plt.cm.gray)

    plt.show()

#show_class_percentage('/dl_data/raw/', 0, 7)
# show_random_cases('/dl_data/raw/', 3, 6, number = 3, has_tumor= True)