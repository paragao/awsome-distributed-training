#!/bin/bash

# Node list as a space-separated array
node_list= " "
nodes=$(scontrol show hostname $node_list)

# Loop through each node and restart slurmd
for node in $nodes; do
    echo "Restarting slurmd on $node"
    ssh $node "sudo systemctl restart slurmd"
done
