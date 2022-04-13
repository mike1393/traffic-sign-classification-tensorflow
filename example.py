#Third-part
#built-in
import os
#local
from utils import train_val_split, order_test_set, create_if_not_found

if __name__ == "__main__":
    path_to_data = os.path.join(os.getcwd(), "data")
    
    if True:
        path_to_train = os.path.join(path_to_data,"Train")
        path_to_new_train = os.path.join(os.path.join(path_to_data,"training_data"), "train")
        path_to_new_val = os.path.join(os.path.join(path_to_data,"training_data"), "val")
        create_if_not_found(path_to_new_train)
        create_if_not_found(path_to_new_val)
        train_val_split(path_to_train=path_to_train, path_to_new_train=path_to_new_train, path_to_new_val=path_to_new_val)
    if False:
        path_to_test = os.path.join(path_to_data, "Test")
        path_to_test_csv = os.path.join(path_to_data,"Test.csv")
        order_test_set(path_to_test=path_to_test, path_to_test_csv=path_to_test_csv)
    
