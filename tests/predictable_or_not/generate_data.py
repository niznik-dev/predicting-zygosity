import argparse
import copy
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("--in_seq_len", type=int, default=5, help="Input sequence length")
args = parser.parse_args()

if args.in_seq_len < 1:
    raise ValueError("Input sequence length must be at least 1.")

# Predictable input and output
pp = []
for i in range(1000):
    line = {'input': '', 'output': ''}
    start = random.randint(0, 1000 - args.in_seq_len - 1)
    for j in range(args.in_seq_len):
        line['input'] += str(start + j) + ','
    line['output'] = str(start + j + 1)
    pp.append(line)

# Predictable input and unpredictable output; just replace the output with a random number
pu = copy.deepcopy(pp)
for line in pu:
    line['output'] = str(random.randint(0, 1000))

# Unpredictable input and predictable output; 5 random numbers and the output is just always 42
up = []
for i in range(1000):
    line = {'input': '', 'output': '42'}
    for j in range(5):
        line['input'] += str(random.randint(0, 1000)) + ','
    up.append(line)

# Unpredictable input and unpredictable output; just random numbers
uu = copy.deepcopy(up)
for line in uu:
    line['output'] = str(random.randint(0, 1000))

# Save the data to JSON files
with open('pp.json', 'w') as f:
    json.dump(pp, f, indent=2)
with open('pu.json', 'w') as f:
    json.dump(pu, f, indent=2)
with open('up.json', 'w') as f:
    json.dump(up, f, indent=2)
with open('uu.json', 'w') as f:
    json.dump(uu, f, indent=2)
