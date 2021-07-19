import numpy as np
import cv2
import os


def sample_image_visualization():
    output_path = 'data/sample_images'
    image_path = 'data/processed.npy'
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


if __name__ == "__main__":
    sample_image_visualization()