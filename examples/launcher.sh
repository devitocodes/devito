#!/bin/bash

# Launch as:
#
#    problem=[acoustic,tti] order=int grid=int mode=[blocked dim1=int dim2=int,original,bench] ./path/to/devito/examples/launcher.sh
#

timestamp=$(date +%F_%T)

export OMP_NUM_THREADS=4
export DEVITO_ARCH=intel
export KMP_AFFINITY=explicit,proclist=[0,1,2,3]

export DEVITO_MIC=0
export DEVITO_PROFILE=1

DEVITO_HOME=/homes/fl1612/Projects/devito
export AT_REPORT_DIR=$DEVITO_HOME/examples/results/devito_at_reports-$problem-$mode-$timestamp
export DEVITO_RESULTS=$DEVITO_HOME/examples/results/devito-results-$problem-$mode-$timestamp
export DEVITO_PLOTS=$DEVITO_HOME/examples/plots/devito-plots-$problem-$mode-$timestamp

if [ -z ${grid+x} ]; then
    grid=256
fi

if [ -z ${order+x} ]; then
    order=2
fi

if [ "$mode" == "bench" ]; then
    space_orders="2 4 6 8 10 12 14 16"
    time_orders="2"
    python $DEVITO_HOME/examples/benchmark.py bench -P $problem -a -o -d $grid $grid $grid -so $space_orders -to $time_orders -r $DEVITO_RESULTS
    python $DEVITO_HOME/examples/benchmark.py plot -P $problem -a -o -d $grid $grid $grid -so $space_orders -to $time_orders -r $DEVITO_RESULTS -p $DEVITO_PLOTS --max_bw 18.2 --max_flops 210
elif [ "$mode" == "blocked" ]; then
    python $DEVITO_HOME/examples/benchmark.py run -P $problem -cb $dim1 $dim2 -o -d $grid $grid $grid -r $DEVITO_RESULTS -so $order -to 2
elif [ "$mode" == "auto" ]; then
    python $DEVITO_HOME/examples/benchmark.py run -P $problem -a -o -d $grid $grid $grid -r $DEVITO_RESULTS -so $order -to 2
else
    python $DEVITO_HOME/examples/benchmark.py run -P $problem -o -d $grid $grid $grid -r $DEVITO_RESULTS -so $order -to 2
fi
