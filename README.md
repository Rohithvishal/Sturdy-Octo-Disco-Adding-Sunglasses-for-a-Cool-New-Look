# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

## Date - 1/2/26
## Reg no - 212224230232

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

##PROGRAM AND OUTPUT:

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
import cv2
import matplotlib.pyplot as plt
faceImage = cv2.imread("Passport photo.jpeg")
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")


```

<img width="347" height="435" alt="download" src="https://github.com/user-attachments/assets/674dfd25-413d-4c3f-b64f-bcc1e937ec47" />



```
faceImage.shape
faceImage.shape
```


(531, 413, 3)

```
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('Image-Source-PlusPNG.com.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```
<img width="552" height="280" alt="download" src="https://github.com/user-attachments/assets/530b6fda-5324-4a1b-a10e-51a67747ede1" />


```
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```
<img width="1209" height="205" alt="download" src="https://github.com/user-attachments/assets/be3e6de2-6267-4a05-82ec-dff5846aa89f" />


```
# Make a copy 
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
h, w, _ = glassBGR.shape   # h = 50, w = 190

y = 160   # eye-level (change only this)
x = 110   # horizontal position

faceWithGlassesNaive[y:y+h, x:x+w] = glassBGR


faceWithGlassesNaive = faceImage.copy()

glassBGR = glassPNG[:, :, :3]   # remove alpha
h, w, _ = glassBGR.shape

y = 160
x = 110

faceWithGlassesNaive[y:y+h, x:x+w] = glassBGR

plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.axis('off')

```
<img width="307" height="389" alt="download" src="https://github.com/user-attachments/assets/3f892a85-2446-4f71-8a5d-d032896f2263" />


```
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
h, w, _ = glassBGR.shape   # h=50, w=190

y = 160   # eye-level (adjust this only)
x = 110   # horizontal position

eyeROI = faceWithGlassesArithmetic[y:y+h, x:x+w]


# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
```
<img width="1597" height="186" alt="download" src="https://github.com/user-attachments/assets/f22f88ce-f419-4af7-a21b-c4a0b45046e4" />


```

# Replace the eye ROI with the output from the previous section
h, w, _ = glassBGR.shape   # glass size

faceWithGlassesArithmetic[y:y+h, x:x+w] = eyeRoiFinal

# Display the final result
# Replace the eye ROI with the output
h, w, _ = glassBGR.shape
faceWithGlassesArithmetic[y:y+h, x:x+w] = eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20])
plt.subplot(121)
plt.imshow(faceImage[..., ::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(faceWithGlassesArithmetic[..., ::-1])
plt.title("With Sunglasses")
plt.axis('off')

```
<img width="1606" height="971" alt="download" src="https://github.com/user-attachments/assets/a6631c25-0087-421d-b6aa-05194d1de788" />


