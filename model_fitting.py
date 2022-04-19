#Third-party
from tensorflow import device
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
#built-in
import os
#local
from utils import create_if_not_found, create_generators, display_history
from models import best_model

def step_scheduler(epoch, lr):
    if epoch > 6:
        lr = 1e-4
    return lr

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
    # callback_list.append(checkpoint_saver)

    # Create callback func to prevent unnecessary trainings
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10)
    callback_list.append(early_stop)

    #Create callback func for LearningRateScheduler
    lr_scheduler = LearningRateScheduler(step_scheduler)
    callback_list.append(lr_scheduler)
    
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
    opt = Adam(epsilon=1e-4)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    with device("/device:GPU:0"):
        history = model.fit(
            train_generator, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callback_list)

    display_history(history, epochs)
