set -x -e
rm -rf rank/exp_rank*
python data/generate_ltr_data.py
python rank/models/train_lgb_ranker.py
python rank/models/train_lgb_ranker.py
python rank/models/serve_lgb_ranker.py
