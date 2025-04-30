#!/usr/bin/env python3

with open('cooperative_vanet/scripts/create_sumo_network.py', 'r') as f:
    content = f.read()

# Replace '<o>' with '<output>'
fixed_content = content.replace('<o>', '<output>')
fixed_content = fixed_content.replace('</o>', '</output>')

with open('cooperative_vanet/scripts/create_sumo_network.py', 'w') as f:
    f.write(fixed_content)

print("File fixed successfully.") 