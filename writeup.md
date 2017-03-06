## Project Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a colour transform and append binned colour features, as well as histograms of colour, to your HOG feature vector.
* Note: for those first two steps don't forget to normalise your features and randomise a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/1.1_car_not_car.png
[image2]: ./output_images/1.2_car_colour_histograms.png
[image3]: ./output_images/1.3_car_hog_visualisation.png
[image4]: ./output_images/2.1_car_feature_normalisation.png
[image5]: ./output_images/2.2_classifier_features.png
[image6.1]: ./output_images/3.1.1_windows.png
[image6.2]: ./output_images/3.1.2_naive_sliding_windows.png
[image6.3]: ./output_images/3.1.3_hog_subsample_sliding_window_search.png
[image7]: ./output_images/3.2.1_problem_classification.png
[image8]: ./output_images/3.2.2_problem_classification_2.png
[image9]: ./output_images/3.3.1_test_image_classification_1.png
[image10]: ./output_images/3.3.2_test_image_classification_2.png
[image11]: ./output_images/3.3.3_test_image_classification_3.png
[image12]: ./output_images/3.3.4_test_image_classification_4.png
[image13]: ./output_images/3.3.5_test_image_classification_5.png
[image14]: ./output_images/3.3.6_test_image_classification_6.png
[image15]: ./output_images/4.1.1_test_video_classification_1.png
[image16]: ./output_images/4.1.2_test_video_classification_2.png
[image17]: ./output_images/4.1.3_test_video_classification_3.png
[image18]: ./output_images/4.1.4_test_video_classification_4.png
[image19]: ./output_images/4.1.5_test_video_classification_5.png
[image20]: ./output_images/4.1.6_test_video_classification_6.png
[image21]: ./output_images/4.1.7_test_video_classification_7.png
[image22]: ./output_images/4.1.8_test_video_classification_8.png
[image23]: ./output_images/4.1.9_test_video_classification_9.png
[image24]: ./output_images/4.1.10_test_video_classification_10.png
[image25]: ./output_images/4.1.11_test_video_classification_11.png
[image26]: ./output_images/4.1.12_test_video_classification_12.png

[video1]: ./test_video_annotated.mp4
[video2]: ./project_video_annotated.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first five code cells of the IPython notebook `notebook.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car - Not Car][image1]

I then explored different colour spaces, histograms of those colour spaces different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like, but have since fixed the indices of the test images to ensure consistent output within the notebook.

Here is an example of the three colour channel histograms.

![Car image colour histograms][image2]

Here is an example using the `RGB` colour space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Image with HOG representation][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that very few combinations of parameters utilised the full range of pixels in the image (evenly dividing into a 64x64 pixel image) and appeared to sufficiently differentiate between the shapes of car and not-car images. I did find that an orientations parameter of 15 worked quite well but produced far too many features for the classifier.

In the end, I settled on these parameters (the same ones used to produce the HOG image above).

- **orientations**: 9
- **pixels_per_cell**: (8,8) primarily because this has sufficient resolution and evenly divides into 64x64.
- **cells_per_block**: (2, 2)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and colour features if you used them).

The code for this step is contained in the 6th through 15th code cells of the IPython notebook `notebook.ipynb`.

I trained a linear SVM using labeled data that consisted of 8792 'cars' and '8968' not cars. Each datum is a 64 pixel by 64 pixel 3 channel image.

The classifier was trained using a train / test split of 20% so the classifier was trained on a random selection of 80% of the data and test accuracy was calculated using the remaining 20% of the data. The data was normalised using SKLearn's `StandardScaler()` preprocessing method. This is an example of that normalisation process using spatial and histogram features.

![Feature normalisation][image4]

Through a series of trial and error tests, I settled on the following features. I considered doing a hyper-parameter search to find the settings that would yield the highest validation accuracy, but extracting the features for all 17760 images was taking roughly 10 minutes so I decided to explore different parameters through trial and error. These final settings were found to produce the highest test accuracy while minimising the number of features. The classifier accuracy was found to be 0.9851 with a total number of features of 6156. Two trials with an additional 2256 features yielded improvements of 0.0042 and 0.0017 in accuracy, but I felt this improvement was trivial compared to an 37% increase in the number of features.

- **Colour Space**: LUV
- **HOG orient**: 9
- **HOG pix_per_cell**: = 8
- **HOG cell_per_block**: = 2
- **HOG channel(s)**: all 3 channels
- **Spatial feature channel(s)**: all 3 channels
- **Spatial Feature dimensions**: (16, 16)
- **Histogram feature channel(s)**: all 3 channels
- **Histogram Feature Bins** = 32

This is a plot showing each of the channels and the features for those channels each for the example car image and the example not-car image.

The first column is the car image channels. The second is the car image features. The third is the not-car channels. The fourth is the not-car image features.

![Image with HOG representation][image5]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the 16th through 19th code cells of the IPython notebook `notebook.ipynb`.

I began by producing an image that includes an array of boxes drawn on the image illustrating the sliding window process. I then attempted a naive sliding window search thats searched the entire image first with 64x64 pixel windows then 96x96 and then 128x128 pixel windows all with 50% overlap. I found this to be very hit and miss with the process finding a large number of false positives (often matching trees as cars).

Windows on the test image.

![Windows][image6.1]

Naive Sliding window search using 96x96 pixel windows with a 50% overlap.

![Naive Sliding window][image6.2]

I then moved on to HOG subsample sliding window search (code cell 18 in the notebook). This process limits the search to only the area where cars would be within the image and calculates the HOG channels of the image before performing the sliding window search. This improves performance as it minimises the cost of producing the HOG channels. Instead of percentage overlap, the overlap is defined as how many cells constitue a slide of the window. The example below was produced using a cell seize of 8x8 pixels and acell step of 2. The window size is always scaled to 64x64 pixels so this step produces a 75% overlap.

There are 4 key steps in the search and classify process:

1. **Sliding Window Search**: Search through the image using a HOG subsample window search.
2. **Heatmap**: Produce a heatmap of the positive matches from the window search.
3. **Labels**: Convert the heatmap into discreet 'labels'. Each label represents a distinct hotspot in the heatmap.
4. **Draw Car Positions**: Draw the bounding box on the image that encompasses the label area.

The following image shows the positive matches for a test image, the resulting heatmap from the positive matches, the 'labels' produced by `scipy.ndimage.measurements.label()`, and the final car positions mapped back onto the original test image. These four images represent the four steps described above.

![Sliding window classification][image6.3]

The search area was defined as the area between 375px and 600px in the y axis and covers all 1280 pixels in the x axis. A scale of 1.5 was used (this is the equivalent of a 96x96 pixel window as a scale of 1 produces a window of 64x64 pixels at the original resolution).

This is the sliding window classification pipeline used for this project. A more complex application of the pipeline that uses various search areas and scales will be discussed next along with how this pipeline is used to analyse the frames of the supplied test and project videos.

#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimise the performance of your classifier?

The code for this step is contained in the 20th through 22nd code cells of the IPython notebook `notebook.ipynb`.

Ultimately I searched using LUV 3-channel HOG features plus spatially binned colour and histograms of colour in the feature vector (the same features described above), which provided a nice result.  

Initial explorations found a few key problems where the classifier was detecting false positives most often near trees, but also where there was a left curve in the yellow lane lines. These two issues are depicted below.

![Problem classification 1][image7]

![Problem classification 2][image8]

I decided to include an analysis of individual bounding boxes and use a few heuristics to determine if the bounding box is 'good', likely to be a car, or 'bad' probably not a car. This was done using the `good_bbox(bbox)` function in the 20th code cell in the notebook.

I considered a bounding box to be bad if:

- The smallest dimension of the box is too small (24 pixels or fewer).
- The box is too narrow (a width less than 75% of the height). I tried a variety of settings for this heuristic, but found the requirement to have a wider box was excluding bounding boxes where only a portion of a car was detected or the car was further in the distance and the box was taller than the car appeared in the image.
- The top of the box was too low on the horizon and smaller than 128 pixels in its smallest dimension.

This proved to be an effective way to minimise false positives but not the only strategy I implemented. When a bounding box is excluded for being 'bad' the pixels still appear in the heatmap. An example of this can be seen in the second test image below.

The next strategy for handling false positives is to utilise the heatmap from the previous 'frame' if one is given. None of the examples below use this strategy as they are non-contiguous, but examples from the test video shown later do use this strategy. The non-zero values of the previous heatmap are set to the heatmap threshold and then added to current frame's heatmap with an opacity of 0.75. This helps 'boost' hotspots that show up in contiguous frames. Limiting previous hotspot values ensures that a bright hotspot from one frame isn't falsely 'remembered' as it diminishes over subsequent frames. I found this to be a useful approach to 'boost' true positives while avoiding 'boosting' false positives.

The last strategy I'll discuss here wasn't implemented to combat false positives, but to improve the performance of the search process within the context of a video. My `detect_vehicles_in_image()` function (in the 20th code cell of the notebook) supports 4 search 'strategies': 'full', 'light', 'limited', and 'skip'.

- **Full (default)**: Search the full 'driveable' area of the image at scales of 0.75, 1.0, 1.5, 2.0, and 2.5. This acts as a detailed search of the image where cars are expected to be detected more 'precisely'. This is the only viable strategy for individual images. The remaining strategies can be used to search video frames with greater performance.
- **Light**: Search the full 'driveable' area of the image but only at a scale of 1.5. This acts as a scan of the image that attempts to find cars 'quickly' accepting that exact locations may not be found, but these locations can be searched in greater detail using the 'limited' search strategy for subsequent searches. When this strategy is used, the hotspot threshold is set to 1 as the 'light' search will inevitably produce fewer overlapping matches.
- **Limited**: Only search within the area of each hotspot of the provided `previous_heatmap` with some padding around the hotspot. The scales used in the search are dependent on the size of the hotspot.
- **Skip**: Use the provided `previous_heatmap` and apply it directly to this image without performing any search or analysis.

The times required to perform these different strategies differ dramatically (hence the need for different strategies). These times are based on the evaluation of the code in the context of iPython notebook on a late 2013 retina macbook pro.

- **Full (default)**: ~ 16.5 seconds
- **Light**: ~ 3 seconds
- **Limited**: ~ 0.8 seconds per hotspot (though it depends on the hotspot size)
- **Skip**: ~ 0.02 seconds

These tests used a 'full' search and a hotspot threshold of 3.

**'Full' search**: 16.45 seconds to analyse image

![Test image classification 1][image9]

**'Full' search**: 16.97 seconds to analyse image

![Test image classification 2][image10]

**'Full' search**: 16.04 seconds to analyse image

![Test image classification 3][image11]

**'Full' search**: 16.84 seconds to analyse image

![Test image classification 4][image12]

**'Full' search**: 16.58 seconds to analyse image

![Test image classification 5][image13]

**'Full' search**: 16.72 seconds to analyse image

![Test image classification 6][image14]

The accuracy of these image is quite good with only the 5th image showing a false positive and partially classified car.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The videos were process in the 23rd code cell of the IPython notebook `notebook.ipynb`.

Here's a [link to an annotated version of the test video][video1]

Here's a [link to my project video result][video2]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the 20th code cell of the IPython notebook `notebook.ipynb`.

The specific details for implementation with respect to performance and false positive exclusion have already been discussed above (in the section where the test images are analysed) so I wont't repeat that here. Instead, I will discuss the specific strategy applied to searching video frames to improve performance.

Here are the first 12 frames of the provided test video showing the original frame, heatmap used to process that frame, the labels produced by `scipy.ndimage.measurements.label()` for that frame, and the car positions drawn back onto the frame.

They were generated using the following video processing settings:

- Light search every 24 frames when a vehicle is being tracked (non-zero values in the heatmap).
- Light search every 4 frames when the heatmap is blank.
- Limited search every 2 frames unless a light search is used.
- Skip the search for every other frame.
- Heatmap threshold of 3.

A 'full' search for every frame would have resulted in an average processing time of roughly 16.5 seconds per frame. The strategy described above resulted in an average frame processing time of 0.85s per frame for the 38 frames. This is a performance improvement of nearly 95% but still no where close to the real time processing of 24 frames per second.

**'Light' search**: 2.87 seconds to analyse image

![Test video frame 1][image15]

**'Skip' search**: 0.02 seconds to analyse image

![Test video frame 2][image16]

**'Limited' search**: 1.6 seconds to analyse image

![Test video frame 3][image17]

**'Skip' search**: 0.02 seconds to analyse image

![Test video frame 4][image18]

**'Limited' search**: 1.44 seconds to analyse image

![Test video frame 5][image19]

**'Skip' search**: 0.02 seconds to analyse image

![Test video frame 6][image20]

**'Limited' search**: 1.62 seconds to analyse image

![Test video frame 7][image21]

**'Skip' search**: 0.02 seconds to analyse image

![Test video frame 8][image22]

**'Limited' search**: 1.68 seconds to analyse image

![Test video frame 9][image23]

**'Skip' search**: 0.02 seconds to analyse image

![Test video frame 10][image24]

**'Limited' search**: 1.63 seconds to analyse image

![Test video frame 11][image25]

**'Skip' search**: 0.02 seconds to analyse image

![Test video frame 12][image26]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started this project by working through problems and getting a simple and somewhat naive classifier and pipeline working and then spent a considerable amount of time refining and improving both the classifier and pipeline.

The two aspects I struggled with most were minimising the detection of false positives and the woeful performance of my initial implementation. The first time I processed the project video, 52 seconds of footage, it took my laptop over 2 hours to complete the annotated video. I felt this was unacceptable so I whittled down my implementation and introduced a variety of search strategies (discussed above) to provide a sufficiently successful pipeline that could handle both individual images and the frames of a video. I think the performance still needs improvement as averaging 0.8 seconds per frame is far from adequate to operate in 'real' time.

An easy way to decrease the average frame processing time is to introduce more 'skip' searches between the 'light' and 'limited' searches, but doing so would quickly impact the quality of the vehicle tracking and would increase the likelihood of a vehicle appearing in frame and not getting tracked right away.

As much as I tried, my implementation still suffers from detecting false positives especially near trees and motorway signs that have strong horizontal and vertical lines within the image. I feel like further work would be required to improve the quality of the classifications. One approach would be to take the problematic frames of the videos, slice up the problem areas into 64x64 pixel slices and perform hard negative mining to improve the accuracy of the classifier within the context of the videos.

An aspect of this project that I think would be interesting to pursue further would be to enhance the pipeline to handle a greater variety of conditions:

- Bad weather conditions as well as sunny.
- A variety of driving conditions (motorway, countryside, mountain, and city driving).
- Day and night driving.

Of course this would require a much bigger collection of labeled data to train the classifier. Udacity have recently released a very large labeled dataset but, from what I understand, that data was collected in Mountain View California, a location that doesn't match all of the conditions above.

That said, the Udacity labeled data includes a larger number of classifications. It would be interesting to enhance the pipeline to support not just 'car' and 'not-car' but also other vehicle classifications.
