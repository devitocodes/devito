#!/bin/bash

touch merge_history.txt

# Current branch tip
git log --pretty=format:'%h' -n 1 >> merge_history.txt

echo ""  >> merge_history.txt

# Last nine PRs merged into master
git log --merges --first-parent master --pretty=format:"%h" -n 9 >> merge_history.txt

echo ""  >> merge_history.txt
