import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D, Dense, Dropout, Activation
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

def best_model(settings: dict):
    """ Create a model by providing parameter settings
    """
    return compare_models(
        number_of_classes=settings["number_of_classes"], 
        conv_block_number=settings["conv_block_number"],
        conv_base_filter=settings["conv_base_filter"],
        dense_size=settings["dense_size"],
        dropout_rate=settings["dropout_rate"],
        batch_norm=settings["batch_norm"],
        double_layer=settings["double_layer"])

def compare_models(
    number_of_classes: int, input_shape=(50,50,3), 
    conv_block_number: int=1, conv_base_filter: int=32,
    dense_size: int=128, dropout_rate: float=.0, 
    batch_norm: bool= False, double_layer: bool=False):
    """ An API for building different model architecture
    """
    input = Input(shape=input_shape)
    for idx in range(conv_block_number):
        if idx==0:
            x = convolution_block(
                input, conv_base_filter, kernal_size=5, dropout_rate=dropout_rate,
                batch_norm=batch_norm, double_layer=double_layer)
        else:
            x = convolution_block(
                x, conv_base_filter, kernal_size=5, dropout_rate=dropout_rate,
                batch_norm=batch_norm, double_layer=double_layer)
        conv_base_filter*=2
    x = GlobalAvgPool2D()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    return Model(inputs=input, outputs=x)

def convolution_block(
    input, number_of_filter: int, kernal_size: int=3, 
    dropout_rate: float=.0, batch_norm: bool= False,
    double_layer: bool=False):
    """Convolution Block
    """
    kernal_size = 3 if double_layer else kernal_size
    x = Conv2D(number_of_filter,kernel_size=kernal_size)(input)
    x = BatchNormalization()(x) if batch_norm else x
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate)(x) if dropout_rate > .0 else x
    if double_layer:
        x = Conv2D(number_of_filter,kernel_size=kernal_size)(x)
        x = BatchNormalization()(x) if batch_norm else x
        x = Activation('relu')(x)
        x = Dropout(rate=dropout_rate)(x) if dropout_rate > .0 else x
    x = MaxPool2D()(x)
    x = BatchNormalization()(x) if batch_norm else x
    return x

def compare_case(test_case: str, idx: int, settings: dict):
    """ Compare models in different control cases

    """
    if test_case == "Convolution Blocks":
        return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=idx+1,
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=settings["batch_norm"],
            double_layer=settings["double_layer"])
    elif test_case == "Convolution Filter":
        return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=pow(2,3+idx),
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=settings["batch_norm"],
            double_layer=settings["double_layer"])
    elif test_case == "Convolution Dropout":
        return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=0.1*idx,
            batch_norm=settings["batch_norm"],
            double_layer=settings["double_layer"])
    elif test_case == "Double Convolution Layer":
        return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=settings["batch_norm"],
            double_layer=idx)
    elif test_case == "Batch Normalization":
        return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=idx,
            double_layer=settings["double_layer"])
    elif test_case == "Advanced":
        #Base Case:
        if idx == 0:
            return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=.0,
            batch_norm=False,
            double_layer=settings["double_layer"])
        # Add BN
        elif idx == 1:
            return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=.0,
            batch_norm=True,
            double_layer=settings["double_layer"])
        # Add BN and Dp
        else:
            return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=True,
            double_layer=settings["double_layer"])

if __name__=="__main__":
    TEST_CASES = {
    "convolution_blocks":("Convolution Blocks",3),
    "convolution_filter":("Convolution Filter",3),
    "conv_dropout":("Convolution Dropout",2),
    "double_layer":("Double Convolution Layer",2),
    "batch_norm":("Batch Normalization",2),
    "advanced":("Advanced",3)}
    # Parameters to updata after each test cases
    settings = {
    "number_of_classes":43,
    "conv_block_number":3,
    "conv_base_filter":32,
    "dense_size":128,
    "dropout_rate":0.1,
    "batch_norm":True,
    "double_layer": True}

    # # Establish Test Cases
    # cases = [
    #     "convolution_blocks", "convolution_filter", 
    #     "conv_dropout", "double_layer","batch_norm","advanced"]
    # test_case, number_of_models = TEST_CASES[cases[5]]
    # print("=================")
    # print(f"Testing case: {test_case}")
    # print("=================")
    # for idx in range(number_of_models):
    #     model = compare_case(test_case, idx, settings)
    #     model.summary()
    model = best_model(settings)
    model.summary()    
