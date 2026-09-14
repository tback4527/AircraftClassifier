# Aircraft Classifier
Aircraft Imagery Classification Resume Project

### Files

model.py contains the PyTorch model

train_aircraft_classifier.py contains the code used to read and process the data, train the classification model, and predict on an unseen dataset.

app.py creates a locally hosted html page that allows a user to predict on a single 20x20 px image and view performance of the full model.

### Model Description

The 32,000 images provided are loaded, enhanced, and turned into arrays before being split into three seperate datasets.
- Training Set = 19,200 images (60%) - Used for training the CNN
- Validation Set = 6,400 images (20%) - Used in assessing and tuning the CNN during training.
- Testing Set = 6,400 images (20%) - Unseen by the CNN during training and used to asses overall performance.

The CNN consists of 9 layers which are described below:
1. 2D Convolution Layer with 32 filters, a 3x3 kernel, and a ReLU activation function
2. 2D Max Pooling Layer that downscales by a factor of 2
3. 2D Convolution Layer with 32 filters, a 3x3 kernel, and a ReLU activation function
4. 2D Max Pooling Layer that downscales by a factor of 2
5. Dropout of 25%
6. Flatten to feed data into dense layers
7. Lazy Linear Layer with 128 units and a ReLU activation function
8. Dropout of 50% 
9. Lazy Linear Layer with 1 unit for classification and a sigmoid activation function

The model uses a Binary Crossentropy loss function and the adam optimizer with default settings.

### Flask Description

Allows the user to upload a 20x20 px image to be classified by the CNN.
The page will then display the image, the original classification if it exists, and the prediction from the CNN.

The page also displays theh confusion matrix and accuracy score from the full CNN's prediction on the testing set and accuracy and loss graphs of the CNN during training.

### To-DO
- Further tune the model by adjusting layer sizes and order, as well as experimenting with optimizer and loss functions.
- Test threshold rate for classifying sigmoid output to find best performing threshold on unseen test set.
- Store images used for data non-locally so dataset can be updated with other datasets and images uploaded to FLASK page.
- Add checks to FLASK app that ensure uploaded image is correct format shape. Attempt to correct shape if not 20x20.
- Clean up FLASK app and create navigation to pages for "User Prediction", "Model Training Results", and "TensorBoard Model Information".