basepath=$(cd `dirname $0`; pwd)
#python tools/prepare_dataset.py --dataset psdb --set train --target ./data/psdb/train.lst --root ${basepath}/../data/psdb
python tools/prepare_dataset.py --dataset psdb --set test --target ./data/psdb/val.lst --shuffle False  --root ${basepath}/../data/psdb

