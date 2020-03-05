import h5py
import json


coco = json.load(open('/data3/wangtan/captioning/self-critical.pytorch/data/cocotalk.json','r'))
feature = h5py.File('/data3/wangtan/openimage/target_dir/coco/coco_train.hdf5', 'r')
new_coco = {}
new_id = []

a = list(feature.keys())
length = len(coco['images'])
for i in range(length):
    print("%d / %d" %(i, length))
    img_id = coco['images'][i]['id']
    if 'tensor(' + str(img_id) + ')' in feature.keys():
        new_id.append(coco['images'][i])

new_coco['images'] = new_id
json.dump(new_coco, open('/data3/wangtan/captioning/self-critical.pytorch/data/cocotalk_new.json','w'))