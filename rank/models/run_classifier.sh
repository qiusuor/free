set -x -e
export SEARCH_END_DAY=21990101
LOG=run_classifer.log
rm -rf $LOG
python /home/qiusuo/free/data/discard.py >>$LOG
python rank/models/train_lgb_classifier.py >>$LOG
python rank/models/train_lgb_classifier.py >>$LOG
python rank/models/serve_lgb_classifier.py >>$LOG

