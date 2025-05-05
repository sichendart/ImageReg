import matplotlib.pyplot as plt
import numpy as np
from image_registration import ImageRegistration as ImgReg

IR = ImgReg()
IR.crop_size = 2

image_path = '/Users/sichen/Box Sync/projects/2024-summer-Sandoval/rad_damage_study'

# file names
# smpl5_f1_name = 'Uncoated_Sample5_Frame1.tif'
# smpl5_f300_name = 'Uncoated_Sample5_Frame300.tif'
smpl6_f1_name = 'Coated_Sample6_Frame1.tif'
smpl6_f900_name = 'Coated_Sample6_Frame900.tif'

# load and crop
IR.load_and_crop_images(image_path+'/'+smpl6_f1_name, image_path+'/'+smpl6_f900_name)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3))
ax0.imshow(IR.base_image, cmap='gray')
ax0.set_title("Sample 6, Frame 1")

ax1.imshow(IR.target_image, cmap='gray')
ax1.set_title("Sample 6, Frame 900")

print('original images infor:', len(IR.base_image.shape), IR.base_image.dtype, np.min(IR.base_image), np.max(IR.base_image))

# normlization
IR.normalize_images()

fig1, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3))
ax0.imshow(IR.base_image_normalized, cmap='gray')
ax0.set_title("Sample 6, Frame 1, norm")

ax1.imshow(IR.target_image_normalized, cmap='gray')
ax1.set_title("Sample 6, Frame 900, norm")

print('normalized images infor:', len(IR.base_image_normalized.shape), IR.base_image_normalized.dtype, np.min(IR.base_image_normalized), np.max(IR.base_image_normalized))

# registration
IR.register_images(method='rigid')
print('shift in x is: ', IR.shift[1])
print('shift in y is: ', IR.shift[0])
print('registered image infor:', len(IR.registered_image.shape), IR.registered_image.dtype, np.min(IR.registered_image), np.max(IR.registered_image))
