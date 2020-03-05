import numpy as np
import h5py
import os

bu_rcnn_dir = '/data4/tingjia/wt/budata/coco_cor2_all_bu'
#box_dir = '/data4/tingjia/wt/budata/cocobu_box'
bu_list = os.listdir(bu_rcnn_dir)

for image in bu_list:
    feature = np.load(os.path.join(bu_rcnn_dir, image))
    #bbox = np.load(os.path.join(box_dir, image))
    image_id = image[:-4]

    with h5py.File('/data4/tingjia/wt/budata/coco_cor2_all_bu.hdf5', 'a') as f:

        image_id_h5py = f.create_group(image_id)
        image_id_h5py.create_dataset("feature", data=feature)
        #image_id_h5py.create_dataset("bbox", data=bbox)

