import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D, Dense, Dropout
from tensorflow.keras.models import Model


def street_signs_model(number_of_classes: int, flatten_method: str= 'avg', input_shape=(40,40,3)):
    input = Input(shape=input_shape)
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

def compare_conv_blocks(
    number_of_classes: int, number_of_blocks: int, 
    input_shape=(40,40,3)):
    input = Input(shape=input_shape)
    for i in range(1,number_of_blocks+1):
        if i==1:
            x = convolution_block(input, 16*i, kernal_size=5)
        else:
            x = convolution_block(x, 16*i, kernal_size=5)
    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    return Model(inputs=input, outputs=x)

def compare_filter_size():
    pass

def compare_dense_size():
    pass

def compare_dropouts():
    pass

def compare_addition_layers():
    pass


def convolution_block(
    input, number_of_filter: int, kernal_size: int=3, 
    dropout_rate: float=0, batch_norm: bool= False):
    x = Conv2D(number_of_filter,kernel_size=kernal_size,activation='relu')(input)
    x = MaxPool2D()(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if dropout_rate > 0:
        x = Dropout(rate=dropout_rate)(x)
    return x

if __name__=="__main__":
    for i in range(3):
        model = compare_conv_blocks(43, i+1)
        model.summary()