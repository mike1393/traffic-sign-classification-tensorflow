import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D, Dense
from tensorflow.keras.models import Model


def street_signs_model(number_of_classes: int, flatten_method: str= 'avg'):
    input = Input(shape=(60,60,3))
    x = convolution_block(input, 32, kernal_size=3)
    x = convolution_block(x, 64, kernal_size=3)
    x = convolution_block(x, 128, kernal_size=3)
    if flatten_method=="flatten":
        x = Flatten()(x)
    else:
        x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    return Model(inputs=input, outputs=x)

def convolution_block(input, number_of_filter: int, kernal_size: int=3):
    x = Conv2D(number_of_filter,kernel_size=kernal_size,activation='relu')(input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    return x

if __name__=="__main__":
    model_with_flatten = street_signs_model(10, flatten_method='flatten')
    model_with_flatten.summary()
    model_with_avg = street_signs_model(10, flatten_method='avg')
    model_with_avg.summary()