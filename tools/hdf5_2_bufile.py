import h5py
import numpy as np

#for i in range(8):
file = h5py.File('/data4/tingjia/wt/budata/coco_vc_xy_openimage_new.hdf5', 'r')
for keys in file:
    feature = file[keys]['feature'][:]
    np.save('/data4/tingjia/wt/budata/coco_vc_xy_openimage_new/'+keys+'.npy', feature)
