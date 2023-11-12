set -x -e
rm -rf rank/exp_cls*
python rank/models/train_lgb_classifier.py
python rank/models/train_lgb_classifier.py
python rank/models/serve_lgb_classifier.py
