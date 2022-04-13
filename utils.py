#Third Party
from sklearn.model_selection import train_test_split
# Built-in
import os
import glob
import shutil
import csv


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

def get_csv_row_number(csv_path: str):
    """ Get the number of data from csv file

    Get the number of data from csv file by reading one row at a time.

    Typical usage example:
    number_of_data = get_csv_row_number("/data/Test.csv")
    
    Args:
        csv_path: The absolute path to test csv folder

    Raises:
        EnvironmentError: An error occurs if the csv.reader cannot open the file
    """
    try:
        data_size=0
        with open(csv_path,'rb') as file_path:
            reader = csv.reader(file_path)
            data_size = sum(1 for row in reader)
    except EnvironmentError:
        print(f"[Error] Cannot open file {csv_path}")
    return data_size

def order_test_set(path_to_test: str, path_to_test_csv: str):
    """ Organizes the testing images into label folders

    Re-organizes the images in test folder into cooresponding label.

    Typical usage example:
    order_test_set("/data/Test", "/data/Test.csv")

    Args:
        path_to_test: The absolute path to test folder
        path_to_test_csv: The absolute path to test csv folder
    Raises:
        EnvironmentError: if the csv.reader cannot open the file
    """
    image_size = get_csv_row_number(path_to_test_csv)-1
    print("Start ordering test files to labeled folder...\n")
    try:
        with open(path_to_test_csv, 'r') as csv_file:
            reader = csv.reader(csv_file,delimiter=",")
            for i,row in enumerate(reader):
                if i==0:
                    continue
                img_name = row[-1][5:]
                img_label = row[-2]
                path_to_folder = os.path.join(path_to_test,img_label)
                create_if_not_found(path_to_folder)
                path_to_image = os.path.join(path_to_test, img_name)
                shutil.move(path_to_image,path_to_folder)
                print(f"Processed Image :[{i}/{image_size}]", end='\r')
            print()
    except EnvironmentError:
        print(f"[Error] Cannot open file {path_to_test_csv}")

