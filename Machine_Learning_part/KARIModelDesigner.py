from AIFactoryDS.AbstractProcesses import ModelDesigner
from KARIPreprocessor import DATASET_TYPE
from typing import Final
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from FasterRCNN import *
import logging
import tensorflow as tf


# model type
class MODEL_TYPE:
    FAST_RCNN: Final = 0
    FASTER_RCNN: Final = 1
    IMPROVED_FAST_RCNN: Final = 2


class KARIModelDesigner(ModelDesigner):
    input_dataset_type = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dataset_type = kwargs.get('dataset_type', DATASET_TYPE.HRSID)
        self.input_shape = kwargs.get('input_shape', (800, 800, 3))
        self.model_type = kwargs.get('model_type', MODEL_TYPE.FASTER_RCNN)


    def build_model(self, **kwargs):
        if self.model is not None:
            logging.info("The model is already built. Loading the model. \n")
            return self.model
        if self.input_dataset_type == DATASET_TYPE.HRSID:
            img_input, shared_layers, roi_input = build_base_model(image_shape=self.input_shape)
            model_rpn, model_classifier, model_all = faster_rcnn(shared_layers, roi_input, img_input)
            self.model_rpn = model_rpn
            self.model_classifier = model_classifier
            self.model = model_all
            return self.model
        elif self.input_dataset_type == DATASET_TYPE.KOMPSAT:
            return

    def loss_function(self, *args, **kwargs):
        pass

    def load_weight(self, **kwargs):
        pass

    def save_weight(self, **kwargs):
        pass

    def __repr__(self):
        return self.representation + "\n"


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[2:], 'GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    kari_model_designer = KARIModelDesigner()
    model_all = kari_model_designer.build_model()
    model_rpn = kari_model_designer.model_rpn
    model_classifier = kari_model_designer.model_classifier
    model_rpn.compile(optimizer='adam', loss=[rpn_loss_cls(9), rpn_loss_regr(9)])
    model_classifier.compile(optimizer='adam',
                             loss=[class_loss_cls, class_loss_regr(38)],
                             metrics={'dense_class_{}'.format(38): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')
    model_all.summary()