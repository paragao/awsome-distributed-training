#!/bin/bash

# Node list as a space-separated array
$NODE_LIST=""
nodes=$(scontrol show hostname "$NODE_LIST")

# Loop through each node and restart slurmd
for node in $nodes; do
    echo "Restarting slurmd on $node"
    ssh -o StrictHostKeyChecking=no$node "sudo systemctl restart slurmd"
done
