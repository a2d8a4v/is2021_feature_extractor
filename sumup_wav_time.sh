#!/usr/bin/env bash

set -euo pipefail

total=0
dir=data/train
each=False

. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

while read -r f; do
    d=$(soxi -D "$f")
    if [[ $each = True ]]; then
        echo "$d s. in $f"
    fi
    total=$(echo "$total + $d" | bc)
done < <(cat $dir/wav.scp | cut -f2 -d" ")

echo "Total : $total seconds"
