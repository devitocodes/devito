#!/bin/bash

# Current branch tip
git log --pretty=format:'%h' -n 1 > merge_history.txt

echo ""  >> merge_history.txt

# Last ten PRs merged into master
git log --merges --first-parent master --pretty=format:"%h" -n 10 >> merge_history.txt
