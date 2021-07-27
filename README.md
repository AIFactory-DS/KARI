# KARI x AI Factory X 에코센싱 - Auomated Moving Target Detection

## Project Structure

```angular2html
└─data
    └─HRSID_JPG
        ├─annotations
        ├─inshore_offshore
        └─JPEGImages
```

- `data` directory contains the sample images and annotaion files
    - `HRSID_JPG` is the reference dataset. There is a download url in their repository. [HRSID](https://github.com/chaozhong2010/HRSID)
## Data Structure
- `HRSID` has a `MS COCO` data format which means a json files have all the information of the dataset.


## References
- [HRSID](https://github.com/chaozhong2010/HRSID)
- [Faster RCNN](https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras)