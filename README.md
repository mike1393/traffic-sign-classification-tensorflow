# Traffic Sign Classification with Tensorflow (98% accuracy)
## :bulb: Project Scope
1. Learn [how to do](https://github.com/mike1393/traffic-sign-classification-tensorflow#mag_right-finding-architecture) a multi-class single-image classification.
2. [Trained](https://github.com/mike1393/traffic-sign-classification-tensorflow#speech_balloon-model-fitting) a neural network to classify German traffic signs.
3. [Evaluate](https://github.com/mike1393/traffic-sign-classification-tensorflow#speech_balloon-model-evaluation) the model.

## :open_file_folder: Dataset
In this project, I am using [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). The script can be found in [./data_cleaning.py](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/data_cleaning.py)
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

## :mag_right: Finding Architecture
Before we start finding the architecture, let's take a look at the base case first. The following strategy is inspired heavily by :fire:[Christ Deotte’s](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook) :fire:work. Please make sure to check it out if you are interested:thumbsup:. Similar to his work, I use the following annotation for model layers.

- Convolution layer: 32C5 denotes Conv2D(filter=32, kernel_size=5, activation=’relu’).
- Max Pooling Layer:  P denotes MaxPool2D().
- Dense layer:  128D denotes Dense(128, activation=’relu’).
- Global Average Pooling layer:  GP denotes GlobalAvgPool2D(). (I used Global Average Pooling to replace the Flatten layer for simplicity since Global Average Pooling prevents overfitting([paper-3.2](https://arxiv.org/pdf/1312.4400.pdf)) and reduces the number of parameters([post](https://stackoverflow.com/a/65860888)))
- Batch Normalization layer: BN denotes BatchNormalization()
- Dropout Layer: 10Dp denotes Dropout(rate=0.1)

The base case for our model can be denoted as the following:

```Input → [ 32C5 → P ] → GP → 128D → 43D(softmax)```

To keep the experiment simple, the following strategies were to find the optimal architecture for the feature extraction layers. i.e.layers between Input and GP.
### :thought_balloon: How many convolution subsampling blocks?
I started off by comparing different numbers of convolution blocks. (Since I set my input size to be (40,40) we cannot do four convolution blocks )
```
1. [ 32C5 - P ]
2. [ 32C5 - P ] → [ 64C5 - P ]
3. [ 32C5 - P ] → [ 64C5 - P ] → [ 128C5 - P ]
```
The result of the comparison can be found below. Since the third case outperforms the rest of the two cases, we now make our architecture: Input → [32C5-P] →[64C5-P]→[128C5-P] → GP→128D→43D(softmax)
![blocks compare](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/conv_block.png)
### :thought_balloon: How many filters?
Now that we know the number of our convolution blocks, we now compare the number of filters. Here are the three cases I’ve tested:
```
1. [ 8C5 - P ] → [ 16C5 - P ] → [ 32C5 - P ]
2. [ 16C5 - P ] → [ 32C5 - P ] → [ 64C5 - P ]
3. [ 32C5 - P ] → [ 64C5 - P ] → [ 128C5 - P ]
```
The result below shows that case three stands out from the test cases. Hence, the architecture is now: Input → [32C5-P] →[64C5-P]→[128C5-P] → GP→128D→43D(softmax)
![filter compare](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/conv_filter.png)
### :thought_balloon: Refinement
After deciding the filter number, the architecture is pretty much done. However, I would like to increase the performance by adding some extra layers.
- Replacing one convolution layer with two consecutive convolution layers<br>
    The first adjustment is to replace one convolution layer with two consecutive convolution layers to increase non-linearity.([[paper](https://arxiv.org/pdf/1409.1556.pdf)], [[post](https://stackoverflow.com/a/51815101)])
    ```
    1. 32C5-P2
    2. 32C3-32C3-P2
    ```
    ![image](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/conv_double_layer.png)
- Prevent Overfitting<br>
    The second adjustment is to prevent model overfitting. I do this by adding batch normalization layers and dropout layers. ([[paper](https://arxiv.org/pdf/1502.03167.pdf)], [[post](https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout)])
    
    Since I am using the Global Average Pooling layer instead of Flatten layer, I am not applying the dropout layer after the Dense layer. However, [[paper](http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf)] and [[post](https://stackoverflow.com/questions/46841362/where-dropout-should-be-inserted-fully-connected-layer-convolutional-layer)] show that adding a small percent of the dropout layer after the convolution layer may increase the performance. Hence, I added an additional dropout layer after batch normalization.
    ```
    1. [ 32C3-32C3-P ]
    2. [ 32C3-BN-32C3-BN-P-BN ]
    3. [ 32C3-BN-DP-32C3-BN-DP-P-BN ]
    ```
    From the result below, we see that the third case performs slightly better than the other two. Hence, until this step, we have decided on our model architecture.
    ![image](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/conv_BN_Dp.png)

## :speech_balloon: Model fitting
The following chart is the training result of our model. I used Adam as my optimizer with epsilon=1e-4(suggested by [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam#args)). I also used a stepped learning rate(lr=0.0001 if epoch > 6 else 0.001) with a learning rate scheduler instead of a constant learning rate(lr=0.001), since I observed oscillation in loss value after the 6th epoch.
![image](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/fitting_result.png)
## :speech_balloon: Model evaluation

After model fitting, I evaluate the model with test data, which results in **98% accuracy**.

Now let’s dig deeper and see what this model is BAD at. Below is the confusion matrix and the classification report on test data.
![image](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/classification_report.png)
We see that the model performs poorly in class 14 and class 20, where it confuses class 14 with class 4 and class 20 with class 10.
![image](https://github.com/mike1393/traffic-sign-classification-tensorflow/blob/main/result/confusion_matrix.png)

