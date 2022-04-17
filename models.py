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

def compare_models(
    number_of_classes: int, input_shape=(40,40,3), 
    conv_block_number: int=1, conv_base_filter: int=32,
    dense_size: int=128, dropout_rate: float=.0, 
    batch_norm: bool= False, double_layer: bool=False):
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
    kernal_size = 3 if double_layer else kernal_size
    x = Conv2D(number_of_filter,kernel_size=kernal_size,activation='relu')(input)
    if double_layer:
        x = Conv2D(number_of_filter,kernel_size=kernal_size,activation='relu')(x)
    x = MaxPool2D()(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if dropout_rate > .0:
        x = Dropout(rate=dropout_rate)(x)
    return x


TEST_CASES = {
    "convolution_blocks":("Convolution Blocks",3),
    "convolution_filter":("Convolution Filter",3),
    "dropout":("Dropout",7),
    "advanced":("Advanced",4)}

def compare_case(test_case: str, idx: int, settings: dict):
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
    elif test_case == "Dropout":
        return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=0.1*idx,
            batch_norm=settings["batch_norm"],
            double_layer=settings["double_layer"])
    elif test_case == "Advanced":
        #Base Case:
        if idx == 0:
            return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=False,
            double_layer=False)
        # Add additional convolution layer
        elif idx == 1:
            return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=False,
            double_layer=True)
        # Add additional Batch Normalization layer
        elif idx == 2:
            return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=True,
            double_layer=False)
        # Add all
        else:
            return compare_models(
            number_of_classes=settings["number_of_classes"], 
            conv_block_number=settings["conv_block_number"],
            conv_base_filter=settings["conv_base_filter"],
            dense_size=settings["dense_size"],
            dropout_rate=settings["dropout_rate"],
            batch_norm=True,
            double_layer=True)

if __name__=="__main__":
    # Parameters to updata after each test cases
    settings = {
    "number_of_classes":43,
    "conv_block_number":3,
    "conv_base_filter":32,
    "dense_size":128,
    "dropout_rate":.0,
    "batch_norm":False,
    "double_layer": False}

    # Establish Test Cases
    cases = ["convolution_blocks", "convolution_filter", "dropout", "Advanced"]
    test_case, number_of_models = TEST_CASES[cases[2]]
    print("=================")
    print(f"Testing case: {test_case}")
    print("=================")
    for idx in range(number_of_models):
        model = compare_case(test_case, idx, settings)
        model.summary()
