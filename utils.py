#Third Party
from sklearn.model_selection import train_test_split
# Built-in
import os
import glob
import shutil


def train_val_split(path_to_data: str, path_to_new_train: str, path_to_new_val: str, split_size: float=0.1):
    """Splits train data for validation

    Split the data into training data and validation data.
    Two new folders will be created containing new training data and new validation data repectively.

    Args:
        path_to_data(str): The path to training data contains GTSRB
        path_to_new_train(str): The path to new training data
        path_to_new_val(str): The path to new validation data
        split_size(float): The percentage of validation data with repect to old training data. Value between [0,1].

    Return:
        None

    """
    # Get the full path of each image and store them into a list
    classification_folders = os.listdir(path_to_data)
    print(f"{len(classification_folders)} class folder(s) found.")
    # Create root folders for train, val if they don't exist
    create_if_not_found(path_to_new_train)
    create_if_not_found(path_to_new_val)
    for folder in classification_folders:
        full_path_of_class = os.path.join(path_to_data, folder)
        image_paths = glob.glob(os.path.join(full_path_of_class, "*.png"))
        # Split the data into train and validate
        train, val = train_test_split(image_paths, test_size=split_size)

        copy_to_folder(train, path_to_new_train, folder)
        copy_to_folder(val, path_to_new_val, folder)

def copy_to_folder(files: str, path: str, folder_id: str):
    """Copies a list of files to target directory

    The path with the class_id will be created if it is not exist
    Each file will be copied to target directory.

    Args:
        files(list(str)): A list of files
        path(str): The path to root folder
        class_id(str): The class id

    Return: None

    """
    folder_path = os.path.join(path, folder_id)
    create_if_not_found(folder_path)
    for file in files:
        shutil.copy(file,folder_path)

def create_if_not_found(path: str):
    """ Recursively create path if not found

    Args:
        path: the path to be found
    Return:
        None
    """
    found = os.path.isdir(path)
    if not found:
        print(f"{path} not found")
        os.makedirs(path)
        print(f"{path} created!")
