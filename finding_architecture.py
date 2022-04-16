#Third-party
from tensorflow import device
from tensorflow.keras.callbacks import EarlyStopping
#built-in
import os
#local
from utils import create_generators, display_performance
from models import compare_conv_blocks

if __name__ == "__main__":
    # Create Generators
    path_to_data = os.path.join(os.getcwd(), "data")
    path_to_train = os.path.join(os.path.join(path_to_data,"training_data"), "train")
    path_to_val = os.path.join(os.path.join(path_to_data,"training_data"), "val")
    path_to_test = os.path.join(path_to_data, "Test")
    batch_size = 64
    epochs = 15
    img_size = (40,40)
    train_generator, val_generator, test_generator = create_generators(
        batch_size,path_to_train,path_to_val,path_to_test,img_size)

    # Create callback functions to run during training
    callback_list=[]

    # Create callback func to prevent unnecessary trainings
    early_stop = EarlyStopping(monitor="val_accuracy", patience=4)
    callback_list.append(early_stop)

    # Create, Compile, and fit the model
    number_of_models = 3
    history=[0]*number_of_models
    names = ["[C-P]","2[C-P]","3[C-P]"]
    number_of_classes = train_generator.num_classes
    line_styles = ['-','--',':','-.']
    for i in range(number_of_models):
        model = compare_conv_blocks(number_of_classes,number_of_blocks=i+1)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        with device("/device:GPU:0"):
            history[i] = model.fit(
                train_generator, 
                batch_size=batch_size, 
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callback_list,
                verbose=2)
            print(f"CNN {names[i]}: Epochs={epochs}, Train accuracy={max(history[i].history['accuracy'])},Validation accuracy={max(history[i].history['val_accuracy'])}")
    display_performance(number_of_models, epochs, history, names, line_styles, ylim=[0.95,1])