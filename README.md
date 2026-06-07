# *Detection and Classification of Radar and Sonar Pulses Using a CNN-LSTM Encoder-Decoder*

* Report is found in `Project`
* All scripts are found in `Src`
* All figures used in the report can be found in `Figures`


## **Packages**
The required packages are
* `numpy`
* `scipy`
* `matplotlib`
* `sklearn`
* `torch`
* `seaborn`


## **Data**
The pipeline starts from a single ocean-noise recording (`coastal_studies_institute_audio_..._CF1E_..._070000Z.wav`), from which the noise segments, spectrograms, and synthetic training examples are derived.

**Note:** `INPUT.npy`, the array of input spectrograms, is **not included** in this repository because the file is too large to upload. It must be generated locally before training or evaluation can be run (see below). The corresponding targets, `TARGET.npy` and `padded_sequences.npy`, are included.


## **Execution**
Run `wav_spectogram.py` to read the `.wav` recording, resample it to 30 kHz, split it into 5 s segments, and z-score normalize each segment, producing `ocean_noise_segments.npy`.

Run `pulses.py` to generate the synthetic dataset. It injects synthetic CW, LFM, and HFM pulses, with modeled transmission loss and reverberation, into the noise segments and writes `INPUT.npy` (the spectrograms) and `TARGET.npy` (the pulse labels). **This step must be run to produce `INPUT.npy`, which is not provided.** A fixed random seed is used, so the dataset is reproduced exactly.

Run `train_model.py` to train the CNN-LSTM encoder-decoder. It reads `INPUT.npy` and `padded_sequences.npy`, trains for 50 epochs, and saves the best checkpoint (by validation loss) along with `train_losses.npy` and `val_losses.npy`.

Run `evaluate.py` to evaluate the trained model on the test set and generate all result figures (detection, classification, regression, and pulse-count). `plot_examples.py` produces the example spectrogram and noise spectrogram shown in the report.

The remaining files are used as libraries. `model.py` contains the network architecture, the forward pass, and the masked loss. `functions.py` and `metrics_core.py` contain helper routines and the evaluation metric. `plots.py` contains the plotting functions called during evaluation.

`best_model_epoch_20_valloss_0.3393.pth` is the trained checkpoint used for the results in the report. It is the model at epoch 20, the validation-loss minimum, with the architecture described in the report (a six-block CNN encoder, a two-layer bidirectional LSTM encoder, and a two-layer LSTM decoder with classification and regression heads).