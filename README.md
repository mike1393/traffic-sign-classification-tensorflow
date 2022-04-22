# Traffic Sign Classification with Tensorflow
## Project Scope
1. Learn how to do a multi-class single-image classification.
2. Trained a neural network to classify German traffic signs.
3. Evaluate the model.

## Dataset
In this project, I am using [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- About the dataset:
    - More than 40 classes
    - More than 50,000 images in total
    - Large, lifelike database
- Things different than MNIST
    1. Images comes in different folders, one class per folder in terms of training folder
    2. Images come in different sizes.
- Data Preprocessing
    1. Folder Cleaning: I split the original training data into a new training data folder and a new validation data folder. In these folders, each image was placed into their correspondence class folder. The testing folder was handled in a similar fashion except for the data splitting.
    2. Image Data Generator: I used the TensorFlow image data generator to rescale and augment the dataset.

## Finding Architecture
Before we start finding the architecture, let's take a look at the base case first. The following strategy is inspired heavily by [Christ Deotte’s](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook) work. Please make sure to check it out if you are interested. Similar to his work, I use the following annotation for model layers.

- Convolution layer: 32C5 denotes Conv2D(filter=32, kernel_size=5, activation=’relu’).
- Max Pooling Layer:  P denotes MaxPool2D().
- Dense layer:  128D denotes Dense(128, activation=’relu’).
- Global Average Pooling layer:  GP denotes GlobalAvgPool2D(). (I used Global Average Pooling to replace the Flatten layer for simplicity since Global Average Pooling prevents overfitting([paper-3.2](https://arxiv.org/pdf/1312.4400.pdf)) and reduces the number of parameters([post](https://stackoverflow.com/a/65860888)))
- Batch Normalization layer: BN denotes BatchNormalization()
- Dropout Layer: 10Dp denotes Dropout(rate=0.1)

The base case for our model can be denoted as the following:

```Input → [ 32C5 → P ] → GP → 128D → 43D(softmax)```

To keep the experiment simple, the following strategies were to find the optimal architecture for the feature extraction layers. i.e.layers between Input and GP.
### How many convolution subsampling blocks?
I started off by comparing different numbers of convolution blocks. (Since I set my input size to be (40,40) we cannot do four convolution blocks )
```
1. [ 32C5 - P ]
2. [ 32C5 - P ] → [ 64C5 - P ]
3. [ 32C5 - P ] → [ 64C5 - P ] → [ 128C5 - P ]
```
The result of the comparison can be found below. Since the third case outperforms the rest of the two cases, we now make our architecture: Input → [32C5-P] →[64C5-P]→[128C5-P] → GP→128D→43D(softmax)
![blocks compare](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/conv_block.png)
