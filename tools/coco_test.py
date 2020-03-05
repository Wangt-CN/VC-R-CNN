import numpy as np
import h5py
import os

test_data = h5py.File('/data1/kaihua/projects/vqa-counting/vqa-v2/genome-test.h5','r')
for i, box in enumerate(test_data['boxes']):
    print(i)
    id = test_data['ids'][i]
    obj_mask = (box.sum(0) > 0).astype(int)
    mask = np.sum(obj_mask)
    box_save = box[:,:mask].T
    np.save('/data4/tingjia/wt/budata/cocobu_box_test/' + str(id) + '.npy', box_save)

# list_dir = os.listdir('/data4/tingjia/wt/budata')
# new_list = []
# for file in list_dir:
#     if 'cocobu_box_test' in file and 'npy' in file:
#         new_list.append(file)
#
# for i in new_list:
#     os.remove('/data4/tingjia/wt/budata/' + i)
