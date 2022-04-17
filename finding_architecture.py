#Third-party
from tensorflow import device
from tensorflow.keras.callbacks import EarlyStopping
#built-in
import os
#local
from utils import create_generators, display_performance
from models import compare_case, TEST_CASES


if __name__ == "__main__":
    # Create Generators
    path_to_data = os.path.join(os.getcwd(), "data")
    path_to_train = os.path.join(os.path.join(path_to_data,"training_data"), "train")
    path_to_val = os.path.join(os.path.join(path_to_data,"training_data"), "val")
    path_to_test = os.path.join(path_to_data, "Test")
    batch_size = 64
    epochs = 2
    img_size = (40,40)
    train_generator, val_generator, test_generator = create_generators(
        batch_size,path_to_train,path_to_val,path_to_test,img_size)
    number_of_classes = train_generator.num_classes
    # Create callback functions to run during training
    callback_list=[]

    # Create callback func to prevent unnecessary trainings
    early_stop = EarlyStopping(monitor="val_accuracy", patience=4)
    callback_list.append(early_stop)



    # Parameters to updata after each test cases
    settings = {
    "number_of_classes":number_of_classes,
    "conv_block_number":3,
    "conv_base_filter":32,
    "dense_size":128,
    "dropout_rate":.0,
    "batch_norm":False,
    "double_layer": False}
    names = ["0%","10%","20%","30%","40%","50%","60%"]

    # Establish Test Cases
    case = 2
    cases = ["convolution_blocks", "convolution_filter", "dropout", "Advanced"]
    titles = ["Block Number", "Filter Number", "Dropout Percentage", "Advanced Layer"]
    test_case, number_of_models = TEST_CASES[cases[case]]
    history=[0]*number_of_models
    print("=================")
    print(f"Number of Models: {number_of_models}")
    print(f"Number of Epochs per model: {epochs}")
    print(f"Test case: {titles[case]} GO!")
    print("=================")
    for i in range(number_of_models):
        model = compare_case(test_case, i, settings)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # Model Training
        with device("/device:GPU:0"):
            history[i] = model.fit(
                train_generator, 
                batch_size=batch_size, 
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callback_list,
                verbose=2)
            print(f"CNN {names[i]}: Epochs={epochs}, Train accuracy={max(history[i].history['accuracy'])},Validation accuracy={max(history[i].history['val_accuracy'])}")
    print("=================")
    print(f"Test case: {titles[case]} Done!")
    print("=================")
    # Plot Result
    display_performance(
        fig_title=titles[case],
        number_of_nets=number_of_models, 
        epochs=epochs, 
        history=history, 
        names=names, 
        ylim=[0.9,1])