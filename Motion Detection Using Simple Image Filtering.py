from PIL import Image
from PIL.Image import fromarray
from os.path import join, exists
from os import listdir, makedirs
import numpy as np
from numpy import zeros, abs, uint8, array, linalg
from scipy import signal
import time # To see time
import pandas as pd

# Globals
image_dir = [r'.\Office', '.\RedChair'] # Input Folder of images
num_of_frames = [100, 352]
alpha = 0.3
noise_output = {1: list(), 2: list(), 3: list(), 4: list(), 5: list()}
out_dir = r'.\color_output'
temporal_deriv_out = r'.\temporal_deriv_out'
gaussian1d_out = r'.\1d_gaussian_out' # Output directory for 1D Gaussian
gaussian2d_out = r'.\2d_gaussian_out' # Output directory for 2D Gaussian
box_out = r'.\boxFilter_out' # Output directory for Box Filter
tune_out = r'.\tune_out'

def norm(pic):
    return pic/255

def get_next_image(i):
    i = i + 1
    images = listdir(image_dir[folder])
    img = Image.open(join(image_dir[folder], images[i]))
    return i, img

""" #Manually convolve but it is too slow to implement, opted to use signal.convolve
def imfilter(img, filter):
result = np.zeros_like(img)
for i in range(1, img.shape[0] - 1):
for j in range(img.shape[1] - len(filter) + 1):
window = img[i-1:i+2, j:j+len(filter)]
result[i, j + 1] = np.sum(window * filter) #Iterates the operator window over every pixel
return result
"""

def motion_detect(i, method, threshold, sigma=None, boxSize=None):
    i, prev_frame = get_next_image(i)
    prev_frame = np.array(prev_frame.convert("L"))

    for frame in range(1, num_of_frames[folder]):
        i, curr_frame = get_next_image(i) # Updating the current frame to the next
        gray_frame = np.array(curr_frame.convert("L")) # Convert current frame into grayscale
        if method == 1:
            deriv = abs(norm(gray_frame) - norm(prev_frame))
            noise_output[method].append([threshold,
            deriv[deriv > threshold].std()])
            output = out_dir # Output directory for Differential Operator
        elif method == 2:
            deriv = temporal_derivative(gray_frame, prev_frame)
            noise_output[method].append([threshold,
            deriv[deriv > threshold].std()])
            output = temporal_deriv_out # Output directory for Temporal Derive 0.5 * [-1 0 1]
        elif method == 3:
            dimension = 1
            deriv = gaussianfilter(gray_frame,
            prev_frame,
            dimension,
            sigma)
            noise_output[method].append([sigma,
            threshold,
            deriv[deriv > threshold].std()])
            output = gaussian1d_out # Output directory for 1D Gaussian Filter
        elif method == 4:
            dimension = 2
            deriv = gaussianfilter(gray_frame,
            prev_frame,
            dimension,
            sigma,
            boxSize)
            noise_output[method].append([boxSize,
            sigma,
            threshold,
            deriv[deriv > threshold].std()])
            output = gaussian2d_out # Output directory for 2D Gaussian Filter
        elif method == 5:
            deriv = boxFilter(gray_frame, prev_frame, boxSize)
            noise_output[method].append([boxSize,
            threshold,
            deriv[deriv > threshold].std()])
            output = box_out # Output directory for Box Filter
        else: # Catch case
            print('Please select a valid method (1-5)')
            return
        
        mask = zeros(deriv.shape, dtype=uint8) # Initializing the mask
        mask[deriv > threshold] = 1 # If the delta is significant (threshold) change that pixel value of 11
        color_mask = zeros((curr_frame.height,
                            curr_frame.width, 3),
                            dtype=uint8) # Initializing a color mask
        
        color_mask[:, :, 1] = mask[0:color_mask.shape[0],
                                    0:color_mask.shape[1]] * 255 # Changing the R G B value to 255 to highlight comp_image = Image.blend(curr_frame,
        
        comp_image = Image.blend(curr_frame,
                        fromarray(color_mask),
                        alpha=alpha) # Concatenate the mask with the original image

        if folder == 0:
            folder_dir = join(tune_out, 'office')
        else:
            folder_dir = join(tune_out, 'redchair')
            method_dir = join(folder_dir, '%s_method' % method)
            boxsize_dir = join(method_dir, '%s_box_size' % boxSize)
            sigma_dir = join(boxsize_dir, '%s_sigma' % sigma)
            
        if not exists(method_dir):
            makedirs(method_dir)
        if not exists(boxsize_dir):
            makedirs(boxsize_dir)
        if not exists(sigma_dir):
            makedirs(sigma_dir)
                
        comp_image.save(join(sigma_dir,
            '{}_threshold (out01_{}).png'.format(threshold,
            str(i).zfill(4)))) # Save and output image
            
        prev_frame = gray_frame
        return
    return

def temporal_derivative(gray_frame, prev_frame):
    T_deriv = np.array([-0.5,0,0.5]).reshape(-1,1) # Temporal Derivative 0.5* [-1 0 1] filter
    deriv = signal.convolve(abs(norm(gray_frame)- norm(prev_frame)),
    T_deriv, mode='same')
    return deriv

def gaussianfilter(gray_frame, prev_frame, dimension, sigma, boxSize = None):
    size = 3 * sigma
    
    if dimension == 1:
        x = np.arange(-size//2+1, size//2 + 1) # Create a 1D mask
        gaussian1d = np.exp(-(x ** 2/(2 * (sigma ** 2)))) # 2D Gaussian Equation from the slides
        gaussian1d = gaussian1d * (-x / sigma ** 2)
        gaussian1d /= np.sum(gaussian1d)
        deriv = signal.convolve(abs(norm(gray_frame) - norm(prev_frame)),
        gaussian1d.reshape(-1,1), mode='same')

    elif dimension == 2:
        center = boxSize // 2
        X,Y = np.mgrid[-center:center+1, -center:center+1] # Create 2-D Gaussian Mask to smooth
        gaussian2d = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) # 2D Gaussian Equation from the slides
        gaussian2d = gaussian2d * ((X * Y) / sigma ** 4)
        gaussian2d /= np.sum(gaussian2d)
        padded_image= np.pad(
                            gray_frame,
                            boxSize//2,
                            mode ='edge') # Padding of the gray_frame to fit the gaussian smoothing mask of boxSize x boxSize
        padded_previmage= np.pad(prev_frame,
                                    boxSize//2,
                                    mode ='edge') # Padding of the prev_frame
        smoothed_image = signal.convolve2d(padded_image,
                                    gaussian2d,
                                    mode='valid') # Create a new smoothed gray_frame as smoothed_smoothed_prev_frame = signal.convolve2d(padded_previmage,
        smoothed_prev_frame = signal.convolve2d(padded_previmage,
                                    gaussian2d,
                                    mode='valid') # Create a new smoothed_prev_frame as smoothed_return temporal_derivative(smoothed_image,smoothed_prev_frame)
        return temporal_derivative(smoothed_image,smoothed_prev_frame)
    else:
        return
    return deriv

def boxFilter(gray_frame, prev_frame, boxSize):
    boxFilt = (1/(boxSize * boxSize)) * np.ones((boxSize,boxSize)) # Zero bias Mean Box filter
    #print(boxFilt)
    padded_image= np.pad(
                        gray_frame,
                        boxSize//2,
                        mode ='edge') # Padding of the gray_frame to fit the gaussian smoothing mask of boxSize x boxSize
    
    padded_previmage= np.pad(
                            prev_frame,
                            boxSize//2,
                            mode ='edge') # Padding of the prev_frame
    
    smoothed_image = signal.convolve2d(
                                        padded_image,
                                        boxFilt,
                                        mode='valid') # Create a new smoothed gray_frame as smoothed_image
    
    smoothed_prev_frame = signal.convolve2d(
                                            padded_previmage,
                                            boxFilt,
                                            mode='valid') # Create a new smoothed_prev_frame as smoothed_image
    
    return temporal_derivative(smoothed_image,
                                smoothed_prev_frame) # Send the smoothed images to find the deltas
    
    
def switchFolder():
    global folder
    folder += 1
    return

def record_results(filename: str, results: dict, columns: list):
    with pd.ExcelWriter(filename) as writer:
        for i, tab in enumerate(list(results.keys())):
            df = pd.DataFrame(results[tab], columns=columns[i])
            df.to_excel(writer, sheet_name=str(tab), index=False)

if __name__ == '__main__':
    filenames = ['office.xlsx', 'red_chair.xlsx']
    column_names = [['Threshold', 'STD'],
    ['Threshold', 'STD'],
    ['Sigma', 'Threshold', 'STD'],
    ['Box_Size', 'Sigma', 'Threshold', 'STD'],
    ['Box_Size', 'Threshold', 'STD']]
    folder = 0
    i = 58
    start_time = time.time()
    for f in range(0, 2):
        for m in range(1, 6):
            if m != 4:
                k_max = 101
            else:
                k_max = 31
            if m in (3, 4):
                l_max = 21
            else:
                l_max = 2
            if m in (4, 5):
                j_max = 21
            else:
                j_max = 2
            for j in range(1, j_max):
                for l in range(1, l_max):
                    for k in range(1, k_max):
                        if j % 2 == 1 or m not in (4, 5):
                            print('folder:', f,
                                    ', method:', m,
                                    ', threshold:', k / 1000,
                                    ", sigma:", l,
                                    ", box size:", j)
                            motion_detect(i, method=m,
                                    threshold=k / 1000,
                                    sigma=l,
                                    boxSize=j)

        record_results(filenames[f], noise_output, column_names)
        switchFolder()
        noise_output = {1: list(), 2: list(), 3: list(), 4: list(), 5: list()}
    print('Program finished in %s seconds' % (time.time() - start_time))
