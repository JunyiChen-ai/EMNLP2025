This repo contains source code for paper 'Equal Truth: Rumor Detection with Invariant Group Fairness'

first, please make sure the structure of this project looks like this:

(display the tree structure of the folder)

To reproduce the experiment results, pelase run the following:
cd FIRM
python FakeNewsDetection/train_invreg.py --hyperparams FakeNewsDetection/best_params_ch.json --language ch --data_path ./processed_data/Chinese_preprocessed_endef.pkl #run experiment on chinese dataset
FakeNewsDetection/train_invreg.py --hyperparams FakeNewsDetection/best_params_en.json --language en #run experiment for english dataset