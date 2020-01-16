import os
import shutil

old_dir = '/home/charlesxujl/data/train/labels/'
new_dir = '/home/charlesxujl/data/test/labels/'

dirs = sorted(os.listdir(old_dir))

for i, f in enumerate(dirs):
    if i >= 34427 // 10:
       break
    print(f)
    shutil.move(os.path.join(old_dir, f), os.path.join(new_dir, f))

