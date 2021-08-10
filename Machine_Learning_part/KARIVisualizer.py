import numpy as np
import cv2
import os
import random


def visualize_sample_image():
    output_path = '../data/sample_images'
    image_path = '../data/processed.npy'
    image_list = np.load(image_path, allow_pickle=True)
    image_list = image_list[()]['train']
    image_name_list = list(image_list.keys())[:5]
    for image_name in image_name_list:
        image_info = image_list[image_name]
        image = image_info['image']
        bboxes = image_info['bbox']
        for bbox in bboxes:
            x, y, w, h = bbox
            x1_y1 = tuple([int(x), int(y)])
            x2_y2 = tuple([int(x + w), int(y + h)])
            cv2.rectangle(image, x1_y1, x2_y2, color=(0, 255, 255), thickness=2, lineType=None, shift=None)
        cv2.waitKey()
        cv2.imshow("image", image)
        cv2.imwrite(os.path.join(output_path, image_name), image)


def visualize_anchor_box(image_size=(150, 150), anchor_scale=[4, 8, 16, 32, 64], anchor_ratio=[1, 1/3, 3]):
    from FasterRCNN import generate_base_anchor_boxes
    anchor_boxes = generate_base_anchor_boxes(image_size=image_size, anchor_ratio=anchor_ratio, anchor_scale=anchor_scale)
    import numpy as np
    blank_image = np.zeros((image_size[0], image_size[1]), np.uint8)
    cx = image_size[0]/2
    cy = image_size[1]/2
    for ib, box in enumerate(anchor_boxes):
        x1 = int(cx+box[0]*image_size[0])
        x2 = int(cx+box[2]*image_size[0])
        y1 = int(cy+box[1]*image_size[1])
        y2 = int(cy+box[3]*image_size[1])
        blank_image = cv2.rectangle(blank_image, (x1, y1), (x2, y2), (int(ib//3)*100, int(ib//3)*100, 255), 1)
    cv2.imshow("base anchor box", blank_image)
    cv2.waitKey(0)


def visualize_sample_image():
    from FasterRCNN import generate_base_anchor_boxes
    from AIFactoryDS.ImageUtilities import iou_x1_y1_x2_y2
    base_anchor_boxes = generate_base_anchor_boxes(image_size=(800, 800),
                                                   anchor_scale=[4, 8, 16, 32, 64],
                                                   anchor_ratio=[1, 1/3, 3])
    image_path = "../data/HRSID_JPG/JPEGImages/P0001_0_800_7200_8000.jpg"
    sample_image = cv2.imread(image_path)
    original_image_size = sample_image.shape
    image_size = (800, 800)
    sample_image = cv2.resize(dsize=image_size, src=sample_image)
    padding_x = 50
    padding_y = 50
    image = cv2.copyMakeBorder(sample_image, padding_x, padding_y, padding_x, padding_y, 0)
    scale_x, scale_y = image_size[0]/original_image_size[0], image_size[1]/original_image_size[1]
    # image_size = (400, 400)
    bounding_boxes = [[603.0, 556.0, 8.0, 10.0], [465.0, 64.0, 12.0, 34.0],
                      [595.0, 550.0, 8.0, 13.0], [636.0, 557.0, 4.0, 11.0],
                      [583.0, 557.0, 7.0, 13.0], [632.0, 555.0, 5.0, 10.0],
                      [585.0, 530.0, 46.0, 13.0], [561.0, 536.0, 6.0, 12.0],
                      [466.0, 105.0, 12.0, 30.0], [531.0, 421.0, 53.0, 22.0], [759.0, 400.0, 29.0, 25.0]]
    for ib, bounding_box in enumerate(bounding_boxes):
        bounding_box = [(bounding_box[0]+padding_x)*scale_x,
                        (bounding_box[1]+padding_y)*scale_y,
                        (bounding_box[0]+padding_x+bounding_box[2])*scale_x,
                        (bounding_box[1]+padding_y+bounding_box[3])*scale_y]
        bounding_box = [int(bbox_) for bbox_ in bounding_box]
        bounding_boxes[ib] = bounding_box
        image = cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 1)
    overlay = image.copy()
    step = 2
    for cx in range(padding_x+step, image_size[0]+padding_x, step):
        for cy in range(padding_y+step, image_size[1]+padding_y, step):
            for ib, anchor_box in enumerate(base_anchor_boxes):
                x1 = int(cx + anchor_box[0]*image_size[0])
                x2 = int(cx + anchor_box[2]*image_size[0])
                y1 = int(cy + anchor_box[1]*image_size[1])
                y2 = int(cy + anchor_box[3]*image_size[1])
                for bounding_box in bounding_boxes:
                    iou = iou_x1_y1_x2_y2((x1, y1, x2, y2), (bounding_box))
                    if iou > 0.55:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (int(ib//3)*40, int(ib//3)*40, 255), 1)
                        break


    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    cv2.imshow("test", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # visualize_sample_image()
    visualize_sample_image()
    visualize_anchor_box(image_size=(400, 400), anchor_ratio=[1, 2, 1/2], anchor_scale=[32, 64, 128])