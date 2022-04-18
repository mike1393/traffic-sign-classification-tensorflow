#Third-party
from tensorflow import device
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#built-in
import os
#local
from utils import create_if_not_found, create_generators
from models import best_model

if __name__ == "__main__":
    # Create Generators
    path_to_data = os.path.join(os.getcwd(), "data")
    path_to_train = os.path.join(os.path.join(path_to_data,"training_data"), "train")
    path_to_val = os.path.join(os.path.join(path_to_data,"training_data"), "val")
    path_to_test = os.path.join(path_to_data, "Test")
    batch_size = 64
    epochs = 15
    train_generator, val_generator, test_generator = create_generators(batch_size,path_to_train,path_to_val,path_to_test)
    # Create callback functions to run during training
    callback_list=[]
    # Create callback func to save model
    path_to_save_model = os.path.join(os.getcwd(), "saved_model")
    create_if_not_found(path_to_save_model)
    checkpoint_saver = ModelCheckpoint(
        filepath=path_to_save_model,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_freq="epoch",
        verbose=1)
    callback_list.append(checkpoint_saver)

    # Create callback func to prevent unnecessary trainings
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10)
    callback_list.append(checkpoint_saver)
    
    # Create, Compile, and fit the model
    number_of_classes = train_generator.num_classes
    settings = {
        "number_of_classes":number_of_classes,
        "conv_block_number":3,
        "conv_base_filter":32,
        "dense_size":128,
        "dropout_rate":.1,
        "batch_norm":True,
        "double_layer": True}
    model = best_model(settings)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    with device("/device:GPU:0"):
        model.fit(
            train_generator, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callback_list)
