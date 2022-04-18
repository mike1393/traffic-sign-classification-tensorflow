#Third-party
from tensorflow.keras.models import load_model
#built-in
import os
#local
from utils import create_generators

if __name__ == "__main__":
    # Create Generators
    path_to_data = os.path.join(os.getcwd(), "data")
    path_to_train = os.path.join(os.path.join(path_to_data,"training_data"), "train")
    path_to_val = os.path.join(os.path.join(path_to_data,"training_data"), "val")
    path_to_test = os.path.join(path_to_data, "Test")
    batch_size = 64
    epochs = 15
    train_generator, val_generator, test_generator = create_generators(batch_size,path_to_train,path_to_val,path_to_test)
    
    # Test Model
    path_to_save_model = os.path.join(os.getcwd(), "saved_model")
    test_model = load_model(path_to_save_model)
    test_model.summary()

    print("Evaluating Validation Set:")
    test_model.evaluate(val_generator)

    print("Evaluating Test Set")
    test_model.evaluate(test_generator)