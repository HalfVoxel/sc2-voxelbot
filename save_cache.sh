#!/bin/bash
cmd="python3 bot/python/predictor_movement_target.py --save-cache; fish"
echo "A"
tmux new-session -d -s foo "$cmd"
echo "B"
tmux rename-window 'Foo'
tmux select-window -t foo:0
tmux split-window -h "$cmd"
tmux split-window -h "$cmd"
tmux split-window -h "$cmd"
tmux split-window -h "$cmd"
tmux split-window -h "$cmd"
tmux split-window -h "$cmd"
tmux select-layout tiled
tmux -2 attach-session -t foo
