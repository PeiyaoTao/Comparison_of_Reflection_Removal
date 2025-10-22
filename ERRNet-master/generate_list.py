# Ziyan Chen, CS7180, 19th Oct 2025
# This code is based on https://github.com/Vandermode/ERRNet

import os

folder = '/scratch/$USER/datasets/transmission_layer'
output = 'VOC_syn.txt'

files = sorted(os.listdir(folder))

with open(output, 'w') as f:
    for filename in files:
        f.write(filename + '\n')

print(f'Generated {output} with {len(files)} files')