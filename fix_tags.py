#!/usr/bin/env python3

# Read the file
with open('cooperative_vanet/scripts/create_sumo_network.py', 'r') as f:
    lines = f.readlines()

# Edit specific lines (based on line numbers from the cat command)
for i, line in enumerate(lines):
    if "<o>" in line:
        lines[i] = line.replace("<o>", "<output>")
    if "</o>" in line:
        lines[i] = line.replace("</o>", "</output>")

# Write the file back
with open('cooperative_vanet/scripts/create_sumo_network.py', 'w') as f:
    f.writelines(lines)

print("File fixed successfully!") 