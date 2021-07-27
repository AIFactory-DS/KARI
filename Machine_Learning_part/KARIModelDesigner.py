from AIFactoryDS.AbstractProcesses import ModelDesigner
from AIFactoryDS.ImageUtilities import iou_x1_y1_x2_y2
from KARIPreprocessor import DATASET_TYPE
from typing import Final
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model, Sequential
import logging


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
        self.input_shape = kwargs.get('input_shape', (800, 800, 1))
        self.model_type = kwargs.get('model_type', MODEL_TYPE.FAST_RCNN)

    @staticmethod
    def VGG_16(input_shape=(600, 600, 3)):
        vgg_model = Sequential()
        # Block 1
        vgg_model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same', name='block1_conv1'))
        vgg_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        vgg_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        vgg_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        vgg_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        vgg_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        vgg_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        vgg_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        vgg_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        vgg_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        vgg_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))

        return vgg_model

    @staticmethod
    def OutputVGG16(vgg_model=None):
        if vgg_model is None:
            vgg_model = KARIModelDesigner.VGG_16()
        vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        vgg_model.add(Flatten(name='flatten_vgg'))
        vgg_model.add(Dense(units=4096, activation='relu', name='fc1_vgg'))
        vgg_model.add(Dense(units=4096, activation='relu', name='fc2_vgg'))
        vgg_model.add(Dense(units=2, activation='softmax', name='softmax_vgg'))
        return vgg_model

    @staticmethod
    def RPN(base_layers, num_anchors):
        base_layers.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1'))

        x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                         name='rpn_out_class')
        x_regression = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                              name='rpn_out_regress')

        return [x_class, x_regression, base_layers]

    @staticmethod
    def CLASSIFIER(base_layers, input_rois, num_rois, nb_classes=2):
        """Create a classifier layer

        Args:
            base_layers: vgg
            input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
            num_rois: number of rois to be processed in one time (4 in here)
            nb_classes: number of classes (2 here, ship or not ship)

        Returns:
            list(out_class, out_regr)
            out_class: classifier layer output
            out_regr: regression layer output
        """

        input_shape = (num_rois, 7, 7, 512)

        pooling_regions = 7

        # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
        # num_rois (4) 7x7 roi pooling
        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

        # Flatten the convlutional layer and connected to 2 FC and 2 dropout
        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)

        # There are two output layer
        # out_class: softmax acivation function for classify the class name of the object
        # out_regr: linear activation function for bboxes coordinates regression
        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                    name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                                   name='dense_regress_{}'.format(nb_classes))(out)

        return [out_class, out_regr]

    def FASTER_RCNN(self, pool_size, num_rois=4, num_anchors=9, input_layer=None):
        # num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)  # 9
        num_anchors = 9
        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(None, 4))
        shared_layers = self.VGG_16(input_layer=img_input)

        rpn = self.RPN(shared_layers, num_anchors)

        classifier = self.CLASSIFIER(shared_layers, roi_input, num_rois, nb_classes=2)

        model_rpn = Model(img_input, rpn[:2])
        model_classifier = Model([img_input, roi_input], classifier)

        # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
        model_all = Model([img_input, roi_input], rpn[:2] + classifier)
        return [model_rpn, model_classifier, model_all]

    def build_model(self, **kwargs):
        if self.model is not None:
            logging.info("The model is already built. Loading the model. \n")
            return self.model
        if self.input_dataset_type == DATASET_TYPE.HRSID:
            self.model = self.FASTER_RCNN(4)
            self.model[2].summary()
            return self.build_model()
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
    kari_model_designer = KARIModelDesigner()
    vgg_model_summary = True
    frcnn_model_summary = False
    if vgg_model_summary:
        vgg_model = kari_model_designer.OutputVGG16()
        from keras.losses import categorical_crossentropy
        vgg_model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
        vgg_model.build((None, 600, 600, 3))
        vgg_model.summary()
    if frcnn_model_summary:
        frcnn_model_summary = kari_model_designer.FASTER_RCNN()
