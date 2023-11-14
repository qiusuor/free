set -x -e
export SEARCH_END_DAY=22990401
python /home/qiusuo/free/data/discard.py
python rank/models/train_lgb_classifier.py
python rank/models/train_lgb_classifier.py
python rank/models/serve_lgb_classifier.py
