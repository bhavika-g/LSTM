# Calcium to Electrophysiology Prediction

This repository implements a deep neural network for predicting subthreshold electrical activity (whole-cell patch clamp ephys) from calcium imaging traces. The model combines temporal convolutions and stacked bidirectional LSTMs for sequence-to-sequence translation of neuronal activity.

## Architecture 
1D Convolution Layers (SiLU, ELU activations)

10-layer Bidirectional LSTM

Narrowing Conv stack for final signal prediction

Custom MSE + smoothness loss for biologically realistic outputs

##Usage 

1. ``` bash
    git clone https://github.com/bhavika-g/LSTM.git
   ```
2. ``` bash
   pip install -r requirements.txt

   ```
3. Load your data from simultaneous whole cell patch clamp and calcium imaging experiments.

4. ``` bash

   python lstm.py

   ```
