import glob
import os
files = glob.glob('*.pkl')
for file in files:
    os.remove(file)