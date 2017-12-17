import glob
import os

files = glob.glob('../images/*')
for i, f in enumerate(files):
    os.rename(f, '../images/' + str(i) + '.jpg')
