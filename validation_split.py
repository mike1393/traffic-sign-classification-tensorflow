#Third-party
#built-in
import os
#local
from utils import train_val_split, create_if_not_found

if __name__ == "__main__":
    #Get the full path of data folder
    path_to_data = os.path.join(os.getcwd(), "data")
    #Get the full path of origin training folder
    path_to_train = os.path.join(path_to_data,"Train")
    #Create the path of new training folder
    path_to_new_train = os.path.join(os.path.join(path_to_data,"training_data"), "train")
    create_if_not_found(path_to_new_train)
    #Create the path of new validation folder
    path_to_new_val = os.path.join(os.path.join(path_to_data,"training_data"), "val")
    create_if_not_found(path_to_new_val)
    #Split the data from old training folder to new training and validation folder
    train_val_split(path_to_train=path_to_train, path_to_new_train=path_to_new_train, path_to_new_val=path_to_new_val)
    
