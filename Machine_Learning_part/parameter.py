class REFERENCE_PARAMETER:
    image_size = [400, 300]
    anchor_box_size = [64, 128, 256]
    anchor_box_scale = [1, 2, 1/2]
    anchor_step = 16
    learning_rate = 1e-3
    num_classes = 100
    num_anchor_boxes = len(anchor_box_size)*len(anchor_box_scale)


class HRSID_PARAMETER:
    image_size = [800, 800]
    anchor_box_size = [4, 8, 16, 32, 64]
    anchor_box_scale = [1, 2, 1/2]
    anchor_step = 2
    learning_rate = 1e-3
    num_classes = 1
    num_anchor_boxes = len(anchor_box_size)*len(anchor_box_scale)
