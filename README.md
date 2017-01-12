# behavior_cloning

A video of the trained model behavior is available at https://www.youtube.com/watch?v=wadaYqsq4Kw.

## Model

### Solution Design

To solve the problem, I started with a slightly modified VGGNet architecture, as described at http://cs231n.github.io/convolutional-networks/.  The model was highly susceptible to overfitting (i.e. training loss would go down steadily as validation loss started increasing), so I added significant 30% dropout on the convolution layers (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), and 50% on the dense layers.  The model was still overfitting, so I reduced the number of layers and reduced the number of parameters until the model stopped overfitting.  

Lastly, I switch the RELUs to Exponential Linear Units (ELUs) - https://arxiv.org/pdf/1511.07289v1.pdf - which significantly sped up the learning process.

I chose not to use a pre-trained network, because I wanted to use a smaller input image size than many of the pre-trained networks target.

### Model Architecture

My model is a CNN with 3 (conv, dropout, ELU, max pool) layers, and one fully connected layer.  The final model I settled on had the following layers:

* 160x50x3 Input Image
* 7x7 Convolution with 8 Filters
* 30% Dropout
* ELU
* 2x2 Max Pool - (80x25x8)
* 3x3 Convolution with 16 Filters
* 30% Dropout
* ELU
* 2x2 Max Pool - (40x12x16)
* 3x3 Convolution with 32 Filters
* 30% Dropout
* ELU
* 2x2 Max Pool - (20x6x32)
* Fully Connected Layer with 64 Parameters - (64)
* 50% Dropout
* ELU
* Output Value (1)

### Training Dataset and Training Process

I collected two training sets, a hard training set which contained 22,279 images and a gold data set containing 36,369 images.  See https://github.com/jcnnghm/behavior_cloning/tree/master/example_images for example images.  The hard set contained images primarily from going around the more difficult turns on the course, and the gold set contained normal driving and recoveries.  

I preprocessed all images before using them by first resizing them to half their original size, to 160x80.  I then cropped out the bottom 5 pixels which mainly contained the cars hood, and the top 25 pixels which contained non-road background.  I also applied a 3-pixel gaussian blur, since I care more about the overall shape of the objects in the image than individual pixel values.

I trained on left, center, and right images, applying a 0.25 correction to the steering angle for the offset images.  During training, I randomly applied a small amount of noise to images 1/3 of the time, randomly adjusted the brightness 1/3 of the time, and used the plain image 1/3 of the time.  I also flipped the image horizontally correcting the angle 50% of the time.  All of these techniques were used to artificially inflate the size of the dataset and prevent overfitting.

I trained on the images in generated batches of 256, starting with 5 epochs on the hard set, followed by 5 epochs on the gold set.  I used keras sample weights to oversample images with greater steering angles.  Training for additional epochs didn't result in significantly lower validation performance.  I trained in two batches so the model would start with a bias toward making sharper turns.

I was able to achieve low mean-squared-error with this approach (approximately 0.085), but depending on the ininitialization that didn't necessarily translate to good simulator performance.

I added a file, `drivetrain.py`, that I could use to fine-tune the model in real-time.  I used the pygame package to interface an Xbox 360 controller with the simulator, and to take manual control.  When no buttons on the controller are pressed, the model fully controls the car as in `drive.py`.  Holding the A button on the controller takes over steering from the model, and collects the images and manual steering angles to train on.  Every time 32 images were collected, each image, as collected and horizontally flipped, with corresponding steering angles, were fed as a batch into the model for training.  The throttle can be manually controlled with the right trigger, and the B button saves an updated weights file.

This process made it very easy to fine-tune the model performance, since recoveries could be trained as errors happened.  In practice, every time I took manual control, I usually collected enough data to train at least a couple of batches, and the impact of the additional training on the model was immediately apparent. 

Overall, the process I used to train was building an initial model and generating a set of weights with model.py, then fine-tuning the weights with drivetrain.py.

### Future Work

The model would likely perform significantly better with a lot more training data - it'd be interesting to try to use reinforcement learning to collect additional data.  Outside of that, the model didn't incorporate any throttle, velocity, or image history.  The model would likely perform better if it was given information about how the car was moving, such as an image from just before the current one.  I think the model performance could also be potentially improved by including some additional manually processed layers, for example, a canny edge detected layer.  That would potentially change the input image dimension from RGB to something like (R, G, B, Canny), or even something like (R, G, B, Canny, Lane Estimate).  
