# Pipeline Leak Detection via Pressure Signal Anomalies

## What this project does
Pipeline leaks cost the industry billions annually and cause serious environmental damage.
Most commercial detection relies on simple threshold alarms that miss slow leaks and
generate excessive false alarms on normal transients.

This LSTM autoencoder learns normal pressure signal behaviour. When a leak distorts the
signal, reconstruction error spikes and that spike is the alarm. Inspired by US Patent 6,970,808 B2.

## Results
AUROC = 0.964. False alarm rate 3.8% at optimal threshold.

## Key insight
The model never sees a leak during training. It only learns what normal looks like.
This is the power of anomaly detection: no labelled fault data required.

## Tech stack
Python 3.10, PyTorch 2.0, NumPy, SciPy, Matplotlib, scikit-learn, MLflow

## How to run
```
conda activate mlenv
pip install -r requirements.txt
python main.py
```

## Dataset
1500 normal pressure-time series from superimposed sine waves + noise.
500 leak signals with a localised pressure-drop anomaly at a random position.

## Author
Joshua Alagoa
