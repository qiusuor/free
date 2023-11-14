set -x -e
python rank/models/train_lgb_ranker.py
python rank/models/train_lgb_ranker.py
python rank/models/serve_lgb_ranker.py
