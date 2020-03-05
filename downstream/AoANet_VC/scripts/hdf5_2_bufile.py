import h5py
import numpy as np

file = h5py.File('/data2/wt/openimages/vc_feature/1coco_train_all_bu_2.hdf5', 'r')
for keys in file:
    feature = file[keys]['feature'][:]
    np.save('/data2/wt/openimages/vc_feature/coco_vc_all_bu/'+keys+'.npy', feature)
