from AIFactoryDS.AbstractProcesses import Trainer


class KARITrainer(Trainer):
    def load_training_dataset(self, **kwargs):
        pass

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