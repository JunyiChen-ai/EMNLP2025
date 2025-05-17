# FIRM: Equal Truth - Rumor Detection with Invariant Group Fairness

This repository contains the source code for the paper "Equal Truth: Rumor Detection with Invariant Group Fairness" submitted for blind review.

## Project Structure
```
FIRM/
├── FakeNewsDetection/
│   ├── backbones/           # Model architectures
│   ├── configs/             # Configuration files
│   ├── data_utils/          # Data processing utilities
│   ├── utils/               # Helper functions
│   ├── train_invreg.py      # Main training script
│   ├── fake_news_fc.py      # Model implementation
│   
└── processed_data/
    ├── Chinese_preprocessed_endef.pkl  # Preprocessed Chinese dataset
    └── FineFake_preprocessed_endef.pkl # Preprocessed English dataset
```

## Reproducing Experiments

To reproduce the experimental results, run the following commands:

```bash
cd FIRM

# Run experiment on Chinese dataset
python FakeNewsDetection/train.py --hyperparams FakeNewsDetection/best_params_ch.json --language ch --data_path ./processed_data/Chinese_preprocessed_endef.pkl

# Run experiment on English dataset
python FakeNewsDetection/train.py --hyperparams FakeNewsDetection/best_params_en.json --language en --data_path ./processed_data/FineFake_preprocessed_endef.pkl
```

