set -x -e
export SEARCH_END_DAY=21990101
LOG=run_rank.log
rm -rf $LOG
python data/discard.py >>$LOG
python data/prepare_minutes_features.py >>$LOG
python rank/models/train_lgb_ranker.py >>$LOG
python rank/models/train_lgb_ranker.py >>$LOG 
python rank/models/serve_lgb_ranker.py >>$LOG
