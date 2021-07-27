from AIFactoryDS.AbstractProcesses import Trainer
from KARIPreprocessor import KARIPreprocessor
from AIFactoryDS.ImageUtilities import normalize_image


class KARITrainer(Trainer):
    dataset_all = None
    dataset_train = None
    dataset_test = None

    def load_training_dataset(self, **kwargs):
        self.dataset_all = KARIPreprocessor.load_processed_data(processed_data_path='data/processed.npy')[()]
        self.dataset_train = self.dataset_all['train']
        for image_ in self.dataset_train:
            self.dataset_train[image_]['image'] = normalize_image(self.dataset_train[image_]['image'])
        sample_image_name = list(self.dataset_train.keys())[0]
        test = self.dataset_train[sample_image_name]['bbox']
        self.dataset_test = self.dataset_all['test']

    def load_pretrained_weight(self, **kwargs):
        pass

    def load_training_recipe(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def __repr__(self):
        return self.representation


if __name__ == "__main__":
    kari_trainer = KARITrainer()
    print(kari_trainer)
    kari_trainer.load_training_dataset()