import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    #print(path, 1)
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        if '.ipynb_checkpoints' in dirpath:
            continue
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    #print('len(path):', path, len(sorted(images)))
    #print('get_paths_from_images:', path, sorted(images)[:5])
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    # Advanced augmentations for soil CT images (only during training)
    add_gaussian_noise = split == 'train' and random.random() < 0.3
    add_gaussian_blur = split == 'train' and random.random() < 0.2
    adjust_intensity = split == 'train' and random.random() < 0.3

    def _augment(img):
        # Geometric augmentations
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)

        # Intensity augmentations (simulate different scanning conditions)
        if adjust_intensity:
            # Random brightness and contrast adjustment
            alpha = random.uniform(0.85, 1.15)  # Contrast
            beta = random.uniform(-0.1, 0.1)    # Brightness
            img = np.clip(alpha * img + beta, 0, 1)

        # Gaussian noise (simulate scanning noise)
        if add_gaussian_noise:
            noise_std = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, img.shape)
            img = np.clip(img + noise, 0, 1)

        # Gaussian blur (simulate different scanning resolutions)
        if add_gaussian_blur:
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(0.3, 0.8)
            # Apply blur channel-wise for grayscale
            for c in range(img.shape[2]):
                img[:, :, c] = gaussian_filter(img[:, :, c], sigma=sigma)

        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [transform2numpy(img) for img in img_list]
    imgs = augment(imgs, split=split)
    ret_img = [transform2tensor(img, min_max) for img in imgs]
    return ret_img

'''


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    for i in imgs:
        print(i.shape)
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img
'''