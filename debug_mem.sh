#!/bin/bash

export DEVITO_LOGGING=ERROR
export DEVITO_DEVELOP=1

count=0

while true; do
    out=$(gdb --batch \
        -ex "set pagination off" \
        -ex "run" \
        -ex "thread apply all bt full" \
        -ex "set logging enabled off" \
        -ex "set print thread-events off" \
        --args python -m pytest -q -vvsx \
        --disable-warnings --no-header --no-summary \
        -m "not parallel" tests/test_data.py 2>&1)

    status=$?

    if [ $status -ne 0 ]; then
        echo "Failed after $count runs"
        echo "$out"
        exit $status
    fi

    ((count++))
    echo "$count"
done

echo "Failed after $count runs"
