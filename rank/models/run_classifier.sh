set -x -e
export SEARCH_END_DAY=20240111
LOG=run_classifer.log
rm -rf $LOG
python data/discard.py >>$LOG
python rank/models/train_lgb_classifier.py >>$LOG
python rank/models/train_lgb_classifier.py >>$LOG
python rank/models/serve_lgb_classifier.py >>$LOG

