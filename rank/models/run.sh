set -x -e
rm -rf rank/exp*
python rank/models/train_lgb_ranker.py
python rank/models/train_lgb_ranker.py
python rank/models/serve_lgb_ranker.py
