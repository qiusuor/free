set -x -e
rm -rf rank/exp*
python rank/models/train_lgb.py
python rank/models/train_lgb.py
python rank/models/serve_lgb.py
