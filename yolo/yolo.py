import os
import random

def setup_dataset():
    data_path = '../data/HRSID_JPG/JPEGImages/'
    image_list = os.listdir(data_path)
    image_path_list = [os.path.join(data_path, image_name)+'\n' for image_name in image_list]
    print(image_path_list[0])
    random.shuffle(image_path_list)
    num_train = int(len(image_path_list)*0.8)
    num_valid = int(len(image_path_list)*0.1)
    with open('train.txt', 'w') as train_file:
        for image_path in image_path_list[:num_train]:
            train_file.write(image_path)
    with open('valid.txt', 'w') as valid_file:
        for image_path in image_path_list[num_train:num_train+num_valid]:
            valid_file.write(image_path)
    with open('test.txt', 'w') as test_file:
        for image_path in image_path_list[num_train+num_valid:]:
            test_file.write(image_path)

setup_dataset()