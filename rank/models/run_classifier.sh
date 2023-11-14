set -x -e
export SEARCH_END_DAY=20230603
python /home/qiusuo/free/data/discard.py
python rank/models/train_lgb_classifier.py
python rank/models/train_lgb_classifier.py
python rank/models/serve_lgb_classifier.py
