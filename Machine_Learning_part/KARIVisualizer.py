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


if __name__ == "__main__":
    # visualize_sample_image()
    visualize_anchor_box(image_size=(400, 400), anchor_ratio=[1, 2, 1/2], anchor_scale=[32, 64, 128])