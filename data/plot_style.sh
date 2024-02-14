set -ex
python data/discard.py
python data/inject_features.py
python data/inject_labels.py
python data/generate_ltr_data.py
python data/generate_style_leaning_feature.py
python data/plot_style_feedback.py
