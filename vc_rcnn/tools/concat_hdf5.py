import h5py

# You have how many hdf5 file
hdf5_num = 4
f = h5py.File("/data2/tingjia/wt/openimage/target_dir/coco/vc-feature/coco_train_all_vc_xy_10_100.hdf5", 'w')

for i in range(hdf5_num):
    a = h5py.File("/data2/tingjia/wt/openimage/target_dir/coco/vc-feature/coco_train_all_vc_xy_10_100_" + str(i) + ".hdf5", 'r')
    for j in a.keys():
        if j not in f.keys():
            ff = f.create_group(j)
            for k in a[j].keys():
                ff.create_dataset(k, data=a[j][k])

f.close()

