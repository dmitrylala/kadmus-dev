import os
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io



def load_data(path, dirs, file_formats, verbose=True):
    """
    Load dataset from given directory.
    
    Parameters
    ------------------
    path : str
        directory from which dataset will be loaded,
        should have directory 'images' (and maybe 'labels')
    dirs : list of str
        names of directories to be loaded from path
    file_formats : list of str
        formats of files to be loaded corresponding to dirs
    verbose : bool

    Returns
    ------------------
    result : list of lists of np.ndarrays with dtype=np.uint8
        result's elements are lists of images/masks in given directories
    """
    result = []
    for dir_name, file_format in zip(dirs, file_formats):
        loaded = []
        directory = os.path.join(path, dir_name)
        filenames = sorted(filter(lambda x: x.endswith(file_format), list(os.walk(directory))[0][2]))
        
        if file_format == '.tif':
            loaded = io.ImageCollection(directory + '/*' + file_format, load_func=lambda f: io.imread(f, plugin='tifffile')).concatenate()
        elif file_format == '.npy':
            loaded = [np.load(os.path.join(directory, filename), fix_imports=False).astype(np.uint8) 
                        for filename in tqdm(filenames, disable=not verbose)]
        result.append(loaded)
    return tuple(result)



def blend(image, mask, alpha=0.2):
    """
    Blend image and it's mask on the same picture.
    
    Parameters
    ------------------
    image : np.ndarray, dtype=np.uint8
    mask: np.ndarray, dtype=np.int32
    alpha : float
        transparency parameter, 0 <= alpha <= 1
    
    Returns
    ------------------
    blended : np.ndarray, dtype=np.uint8
        result of applying PIL.Image.blend to given parameters
    """
    mask = mask.astype(np.int32)
    zeros = np.zeros_like(mask)
    mask = np.dstack((mask * 255, zeros, zeros)).astype(np.uint8)
    return Image.blend(Image.fromarray(image), Image.fromarray(mask), alpha=alpha)



def show_images(images, titles = []):
    """
    Show a list of images with given titles.
    
    Parameters
    ------------------
    images : list of np.ndarray, dtype=np.uint8
    titles : list of str
        titles of pictures, should have same len as images
    
    Returns
    ------------------
    None
    """
    n = len(images)
    figure = plt.figure(figsize=(15,15))
    for i in range(n):
        figure.add_subplot(1, n, i + 1)
        plt.imshow(images[i])
        if titles != []:
            plt.xlabel(titles[i])
    plt.show(block=True)



def get_blocks(images, block_size, verbose=True):
    """
    Get blocks of square images with given block size.
    
    Parameters
    ------------------
    images : list of np.ndarray
        list of images
    block_size : int
        the size of the blocks into which images will be divided
    
    Returns
    ------------------
    images_blocks : np.ndarray, dtype=np.uint8
        list of all blocks
    """
    images_blocks = []
    pad_width = (block_size - (images[0].shape[0] % block_size)) // 2
    for k in tqdm(range(len(images)), disable=not verbose):
        if len(images[k].shape) == 3:
            image_padded = np.pad(images[k], ((pad_width, pad_width), (pad_width, pad_width), (0, 0)))
        else:
            image_padded = np.pad(images[k], ((pad_width, pad_width), (pad_width, pad_width)))
        n_blocks_height, n_blocks_width = image_padded.shape[0] // block_size, image_padded.shape[1] // block_size
        for j in range(n_blocks_height):
            for i in range(n_blocks_width):
                up, down = j * block_size, (j + 1) * block_size
                left, right = i * block_size, (i + 1) * block_size
                image_block = image_padded[up:down,left:right]
                images_blocks.append(image_block)
    return np.array(images_blocks, dtype=np.uint8)



def get_augs(images, labels, transforms, num_augs):
    """
    Get unique augmentations from 'transforms' composition.
    
    Parameters
    ------------------
    images : list of np.ndarray, dtype=np.uint8
    labels : list of np.ndarray, dtype=np.uint8
    transforms : class albumentations.core.composition.Compose
        composition of transforms applied to images and labels
    num_augs : int
        number of augmentations for each image

    Returns
    ------------------
    transformed_images : list of np.ndarray, dtype=np.float32
        list of images augmentations
    transformed_labels : list of np.ndarray, dtype=np.float32
        list of labels corresponding to images augmentations
    """
    transformed_images, transformed_labels = [], []
    for i in tqdm(range(len(images))):
        transformed_images.append(images[i])
        transformed_labels.append(labels[i])

        cur_transformed_images, cur_transformed_labels = [], []
        for j in range(num_augs):
            transformed = transforms(image=images[i], mask=labels[i])
            is_duplicate = np.any([np.all(transformed['image'] == transformed_image) for transformed_image in cur_transformed_images])
            if not is_duplicate:
                cur_transformed_images.append(transformed['image'])
                cur_transformed_labels.append(transformed['mask'])
        transformed_images += cur_transformed_images
        transformed_labels += cur_transformed_labels
    return np.array(transformed_images, dtype=np.float32), np.array(transformed_labels, dtype=np.float32)



def mask_pipeline(image, model, block_size=256, verbose=False):
    image_blocks = get_blocks([image], block_size, verbose)
    result_blocks = np.round(model.predict(image_blocks.astype(np.float32))).astype(np.uint8)
    
    img_size = image.shape[0]
    pad_width = (block_size - (img_size % block_size)) // 2
    mask_shape = (img_size + 2 * pad_width, img_size + 2 * pad_width)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    height, width = mask_shape
    num = 0
    for j in range(height // block_size):
        for i in range(width // block_size):
            up, down = j * block_size, (j+1) * block_size
            left, right = i * block_size, (i+1) * block_size
            mask[up:down,left:right] = result_blocks[num][:,:,0]
            num += 1
    return mask[pad_width:img_size + pad_width, pad_width:img_size + pad_width]