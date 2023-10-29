import numpy as np
import sys

# run this script as: python3 npy_convert.py <filename>.npy
# writes a .bin file

# File name
filename = sys.argv[1][0:-4]

# Load the weights from the .npy file
weights = np.load(filename + '.npy')

# Flatten the weights
flattened_weights = weights.flatten().astype(np.float32)

# Define the name of the output binary file
output_binary_file = filename + '.bin'

# Save the flattened weights to a binary file
flattened_weights.tofile(output_binary_file)

print(f'Flattened weights have been saved to {output_binary_file}')
