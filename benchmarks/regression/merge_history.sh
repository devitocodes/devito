#!/bin/bash

touch merge_history.txt

# Current branch tip
git log --pretty=format:'%H' -n 1 >> merge_history.txt

echo ""  >> merge_history.txt

# Last nine PRs merged into master
git log --merges --first-parent origin/master --pretty=format:"%H" -n 9 >> merge_history.txt

echo ""  >> merge_history.txt
