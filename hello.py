if __debug__:
    import inspect

import math
import logging
import numpy as np
import os
import matplotlib.pyplot as plt   

from scipy import ndimage as ndi
from scipy import signal
from skimage import data, filters, io, feature, exposure, measure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import rescale

from skimage.exposure import rescale_intensity
import argparse
import cv2
import matplotlib.patches as patches
from PIL import Image


def convolve(image, kernel, cutoff):
    output = signal.convolve2d(image, kernel, boundary='symm', mode='same')

    # Set values between 0 - 1
    max = np.amax(output)
    min = np.amin(output)
    range = max - min
    if min < 0 :
        output = output - min
        max = max - min
        min = 0

    output = (output - min) / range

    # Only keep the extreme findings
    if (cutoff > 0):
        output = np.where(output < cutoff, 0, output)
        output = np.where(output >= cutoff, 1, output)
 
	# return the output image
    return output

def find_top_left_start_candidates(image):
    kernel = np.array((
        [-15,     -15,       -15,      35,      35],
        [-15,     -15,       -15,      35,      35],
        [-15,     -15,       -15,      15,      15],
        [ 35,      35,        15,      15,      15],
        [ 35,      35,        15,      15,      15],
    ), dtype="int")

    image = convolve(image, kernel, 0.8)

    if __debug__:
        filename = os.path.join("./Debug/", inspect.currentframe().f_code.co_name + " - " + str(image.shape[0]) + "x" + str(image.shape[1]) + "TopLeftBlackCornerDetection.jpg")
        io.imsave(filename, image)        
        
    # Detect the top left pixel of those corners
    kernel = np.array((
        [3,     2,      0],
        [2,     1,      -5],
        [0,     -5,      -5],
    ), dtype="int")

    image = convolve(image, kernel, 0.85)

    if __debug__:
        filename = os.path.join("./Debug/", inspect.currentframe().f_code.co_name + " - " + str(image.shape[0]) + "x" + str(image.shape[1]) + "TopLeftPixelStartDetection.jpg")
        io.imsave(filename, image)
        # io.imshow(image)
        

    # Find possible starting locations
    indices = np.where(image == 1)

    # Tuning param, the kernels seem to begin too early
    indices[0][:] = indices[0][:] + 2
    indices[1][:] = indices[1][:] + 1

    return image, indices

def calculate_start_indicator_boxes(image_binary, top_left_start_indices):
    # Attempts to find the start indicator boxes by looking at the diagonal size and start indice
    # Returns matrix containing rows of: StartX, StartY, EndX, EndY, MidX, MidY, Width, Height, IDBitWidth

    maxIndices = 1000
    boxInformation = np.zeros((maxIndices, 9), dtype=np.int32)
    nextIndiceId = 0
    idBitWidthRelation = 0.75

    # Find the initial set of candidates
    for index in range(len(top_left_start_indices[0])):
        rightX = top_left_start_indices[1][index]
        bottomY = top_left_start_indices[0][index]
        diagonalSize = 1
        whiteStepsStart = 3
        whiteStepsRemaining = whiteStepsStart
        cancelSearch = False
        
        while whiteStepsRemaining > 0 and nextIndiceId < maxIndices and cancelSearch == False:
            newX = rightX + 1
            newY = bottomY + 1
            if newX >= image_binary.shape[1] or newY >= image_binary.shape[0]:
                break
            if image_binary[newY, newX]:
                whiteStepsRemaining = whiteStepsRemaining - 1
            else:
                whiteStepsRemaining = whiteStepsStart
            if whiteStepsRemaining == whiteStepsStart:
                rightX = newX
                bottomY = newY
                diagonalSize = diagonalSize + 1
        if diagonalSize > 4 and diagonalSize < 300:
            entry = np.array([
                top_left_start_indices[1][index],
                top_left_start_indices[0][index],
                rightX,
                bottomY,
                top_left_start_indices[1][index] + ((rightX - top_left_start_indices[1][index]) / 2),
                top_left_start_indices[0][index] + ((bottomY - top_left_start_indices[0][index]) / 2),
                rightX - top_left_start_indices[1][index],
                bottomY - top_left_start_indices[0][index],
                (rightX - top_left_start_indices[1][index]) * idBitWidthRelation
            ])
            boxInformation[nextIndiceId] = entry
            nextIndiceId = nextIndiceId + 1

    if nextIndiceId >= maxIndices:
        logging.warning("Maximum start indicator boxes found. Ending search prematurely")

    boxInformation = np.delete(boxInformation, range(nextIndiceId, maxIndices), 0)

    return boxInformation

def mean_pixel_intensity(image, startX, startY, endX, endY):
    # Calculates the mean pixel intensity of a rectangle area inside an image
    width = endX - startX
    height = endY - startY
    return np.sum(image[np.ix_(range(startY, endY), range(startX, endX))]) / (width * height)

def filter_start_indicator_boxes_duplicates(image, start_indicator_boxes):
    # Removes boxes which have close to the same starting position
    filtered_boxes = np.zeros((0, start_indicator_boxes.shape[1]), dtype=np.int32)
    for boxID in range(start_indicator_boxes.shape[0]):
        checkBox = start_indicator_boxes[boxID]
        startX = checkBox[0]
        startY = checkBox[1]
        if np.any(abs(filtered_boxes[np.where(abs(filtered_boxes[:, 0] - startX) < 15), 1] - startY) < 15) == False:
            filtered_boxes = np.append(filtered_boxes, checkBox.reshape((1, start_indicator_boxes.shape[1])), 0)

    return filtered_boxes

def filter_start_indicator_boxes_whitesurround(image, start_indicator_boxes):
    # Tests the start indicator boxes against the expected shape of the start indicator and filters out incorrect candidates
    brightnessCutoff = 0.75
    acceptableIDs = []
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    for boxID in range (start_indicator_boxes.shape[0]):
        idStartX = start_indicator_boxes[boxID, 0]
        idEndX = start_indicator_boxes[boxID, 2]
        idWidth = start_indicator_boxes[boxID, 6]
        midY = start_indicator_boxes[boxID, 5]
        idHeight = start_indicator_boxes[boxID, 7]
        bitWidth = start_indicator_boxes[boxID, 8]
        bitHalfWidth = math.floor(bitWidth * 0.5)
        # Check image size restrictions
        if idEndX + (bitWidth * 13) >= imageWidth or idStartX - (idWidth * 2) < 0 or midY - (idHeight * 3) < 0 or midY + (idHeight * 3) >= imageHeight:
            continue
        # Check white area right of start indicator box
        startX = idEndX + bitHalfWidth - math.floor(bitWidth / 3)
        startY = midY - math.floor(idWidth / 3)
        endX = idEndX + bitHalfWidth + math.floor(bitWidth / 3)
        endY = midY + math.floor(idWidth / 3)
        if mean_pixel_intensity(image, startX, startY, endX, endY) < brightnessCutoff:
            continue
        # Check white area 2x right of start indicator box
        startX = idEndX + bitWidth + bitHalfWidth - math.floor(bitWidth / 3)
        startY = midY - math.floor(idWidth / 3)
        endX = idEndX + bitWidth + bitHalfWidth + math.floor(bitWidth / 3)
        endY = midY + math.floor(idWidth / 3)
        if mean_pixel_intensity(image, startX, startY, endX, endY) < brightnessCutoff:
            continue
        # Check white area left of start indicator box
        startX = idStartX - bitHalfWidth - math.floor(bitWidth / 3)
        startY = midY - math.floor(idWidth / 3)
        endX = idStartX - bitHalfWidth + math.floor(bitWidth / 3)
        endY = midY + math.floor(idWidth / 3)
        if mean_pixel_intensity(image, startX, startY, endX, endY) < brightnessCutoff:
            continue

        acceptableIDs.append(boxID)

    return start_indicator_boxes[acceptableIDs]

def locate_ids(image):
    # The convolution kernels are only good for IDs of a certain size range. Scale down a few times for better results.
    rescales = [1, 0.5, 0.25, 0.125]
    start_indicator_boxes = np.zeros((0, 9))

    for imageScale in rescales:
        if imageScale == 1:
            scaledImage = image
        else:
            scaledImage = rescale(image, imageScale, anti_aliasing=True)
        image_binary = threshold_otsu(scaledImage) < scaledImage
        top_left_image, top_left_start_indices = find_top_left_start_candidates(scaledImage)
        scaled_start_indicator_boxes = calculate_start_indicator_boxes(image_binary, top_left_start_indices)
        start_indicator_boxes = np.append(start_indicator_boxes, np.floor(scaled_start_indicator_boxes / imageScale), 0).astype(int)
        #io.imshow(image_binary)

    # Make all sorts of checks to filter out faulty boxes
    start_indicator_boxes = filter_start_indicator_boxes_duplicates(image, start_indicator_boxes)
    start_indicator_boxes = filter_start_indicator_boxes_whitesurround(image, start_indicator_boxes)

    return start_indicator_boxes

def read_ids(image, start_indicator_boxes):
    # Reads the ID values, filters IDs which do not pass the CRC checks
    for boxID in range (start_indicator_boxes.shape[0]):
        startBox = start_indicator_boxes[boxID]
        boxStartX = startBox[0]
        boxStartY = startBox[1]
        boxEndX = startBox[2]
        boxEndY = startBox[3]
        boxMidX = startBox[4]
        boxMidY = startBox[5]
        boxWidth = startBox[6]
        boxHeight = startBox[7]
        idBitWidth = startBox[8]






        
# Read image
filename = os.path.join("./Samples/", "20190824_105242.jpg")
image_orig = io.imread(filename)
image = image_orig

# Convert to grayscale
image = rgb2gray(image)

# Scale the intensity evenly
v_min, v_max = np.percentile(image, (0.2, 99.8))
image = exposure.rescale_intensity(image, in_range=(v_min, v_max))

id_locations = locate_ids(image)
ids = read_ids(image, id_locations)

image_filtered = image
image_filtered = np.where(image_filtered < 0.20, 0, image_filtered)
image_filtered = np.where(image_filtered > 0.80, 1, image_filtered)
image_filtered = np.where((image_filtered >= 0.20) & (image_filtered <= 0.8), 0.5, image_filtered)

image_binary = threshold_otsu(image) < image

# contours = measure.find_contours(image_binary, 0.1)

# Display results
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax0.imshow(image, cmap=plt.cm.gray)
ax0.axis('off')
ax0.set_title('image', fontsize=10)

#ax1.scatter(top_left_indices[1], top_left_indices[0])
# ax1.axis('off')
    
#ax1.imshow(image_convolved, cmap=plt.cm.gray)
#ax1.axis('off')
#ax1.set_title('image convolved', fontsize=10)

#ax1.set_title('Contours', fontsize=10)
#for n, contour in enumerate(contours):
#    if n < 2000:
#        ax1.plot(contours[n][:, 1], contours[n][:, 0], linewidth=2)

ax2.imshow(image_binary, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Binary image', fontsize=10)

for index in range(id_locations.shape[0]):
    xStart = id_locations[index][0]
    yStart = id_locations[index][1]
    xLength = id_locations[index][2] - id_locations[index][0]
    yLength = id_locations[index][3] - id_locations[index][1]
    rect = patches.Rectangle((xStart,yStart),xLength,yLength,linewidth=1,edgecolor='r',facecolor='none')
    ax2.add_patch(rect)

fig.tight_layout()

plt.show()