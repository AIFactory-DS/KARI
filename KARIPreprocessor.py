import json
import logging
import os
import cv2
import numpy as np
from typing import Final
from AIFactoryDS.AbstractProcesses import Preprocessor


# data type
class DATASET_TYPE:
    HRSID: Final = 0
    KOMPSAT: Final = 1


class KARIPreprocessor(Preprocessor):
    dataset_type = None
    directory_path = None
    dataset = None
    shape = None

    def __init__(self, **kwargs):
        self.dataset_type = kwargs.get('dataset_type', DATASET_TYPE.HRSID)
        self.directory_path = kwargs.get('directory_path', 'data/HRSID_JPG')
        logging.basicConfig(level=kwargs.get('log', logging.INFO))

    @staticmethod
    def open_json(file_path: str):
        json_object = None
        with open(file_path) as f:
            json_object = json.load(f)
        return json_object

    def load_HRSID_dataset(self, **kwargs):
        image_directory = os.path.join(self.directory_path, 'JPEGImages')
        annotation_directory = os.path.join(self.directory_path, 'annotations')
        _ = os.listdir(image_directory)
        image_list = [os.path.join(image_directory, f) for f in _]
        annotation_train_path = os.path.join(annotation_directory, 'train2017.json')
        annotation_test_path = os.path.join(annotation_directory, 'test2017.json')
        annotation_train = self.open_json(annotation_train_path)
        annotation_test = self.open_json(annotation_test_path)

        def extract_boxes(annotations):
            image_id_map = {}
            image_map = {}
            for image_info in annotations['images']:
                image_id_map[image_info['id']] = image_info['file_name']
                image_path = os.path.join(image_directory, image_info['file_name'])
                image_map[image_info['file_name']] = {'image': np.array(cv2.imread(image_path))}

            for anno in annotations['annotations']:
                file_name = image_id_map[anno['image_id']]
                if 'bbox' in image_map[file_name].keys():
                    image_map[file_name]['bbox'].append(anno['bbox'])
                else:
                    image_map[file_name]['bbox'] = [anno['bbox']]
            return image_map
        image_train = extract_boxes(annotation_train)
        image_test = extract_boxes(annotation_test)
        self.dataset = {'train': image_train, 'test': image_test}
        return self.dataset

    def load_original_data(self, **kwargs):
        logging.info("Start loading dataset from directory " + self.directory_path + '.\n')
        if self.dataset is not None:
            return self.dataset
        if self.dataset_type == DATASET_TYPE.HRSID:
            return self.load_HRSID_dataset(**kwargs)
        elif self.dataset_type == DATASET_TYPE.KOMPSAT:
            logging.warning('KOMPSAT dataset loader is not defined yet.')
            return None

    def save_processed_data(self, **kwargs):
        if self.dataset is None:
            logging.warning('You need to load the data before save the dataset. \n' +
                            'Please call `load_original_dataset` first. \n')
            return False
        elif kwargs.get('output_file_path') is None:
            logging.warning('Saving the dataset into a default path: `data/processed.npy` \n')
        np.save(kwargs.get('output_file_path', 'data/processed.npy'), self.dataset, allow_pickle=True)
        return True

    def process(self, **kwargs):
        pass

    def split_dataset(self, **kwargs):
        pass

    def execute(self):
        self.load_original_data()
        self.save_processed_data()

    @staticmethod
    def load_processed_data(**kwargs):
        dataset = np.load(kwargs.get('processed_data_path', 'data/processed.npy'), allow_pickle=True)
        return dataset

    def __repr__(self):
        if len(self.representation) == 0:
            self.representation = ''
            self.representation += 'This processor processes '
            if self.dataset_type == DATASET_TYPE.HRSID:
                self.representation += 'HRSID'
            elif self.dataset_type == DATASET_TYPE.KOMPSAT:
                self.representation += 'KOMPSAT'
            self.representation += ' type of dataset.\n'
            self.representation += 'That is all for now...'
            return self.__repr__()
        else:
            return self.representation


if __name__ == "__main__":
    kari_data_manager = KARIPreprocessor()
    print(kari_data_manager)
    kari_data_manager.load_original_data()
    kari_data_manager.save_processed_data()
