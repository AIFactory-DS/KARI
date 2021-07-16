from AIFactoryDS.AbstractProcesses import ModelDesigner
from KARIPreprocessor import DATASET_TYPE
from typing import Final
from keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
import logging


# model type
class MODEL_TYPE:
    FAST_RCNN: Final = 0
    FASTER_RCNN: Final = 1
    IMPROVED_FAST_RCNN: Final = 2


class KARIModelDesigner(ModelDesigner):
    input_dataset_type = None

    def __init__(self, **kwargs):
        self.input_dataset_type = kwargs.get('dataset_type', DATASET_TYPE.HRSID)
        self.input_shape = kwargs.get('input_shape', (800, 800, 1))
        self.model_type = kwargs.get('model_type', MODEL_TYPE.FAST_RCNN)

    @staticmethod
    def VGG_16(input_layer=None, input_shape=(800, 800, 1)):
        if input_layer is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_layer):
                img_input = Input(tensor=input_layer, shape=input_shape)
            else:
                img_input = input_layer

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    @staticmethod
    def RPN(base_layers, num_anchors):
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
            base_layers)

        x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(
            x)
        x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                        name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]

    def FAST_RCNN(self,  pool_size, num_rois, input_layer=None):
        vgg_model = self.VGG_16(input_layer)


    def build_model(self, **kwargs):
        if self.model is not None:
            logging.info("The model is already built. Loading the model. \n")
            return self.model
        if self.input_dataset_type == DATASET_TYPE.HRSID:
            return self.FRCNN()
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
    kari_model = KARIModelDesigner()
    print(kari_model)