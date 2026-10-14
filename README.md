# Urban Noise Classification on Edge Devices

Masters research code for classifying urban noise-pollution sources in Kampala-style soundscapes using
transfer learning on [YAMNet](https://tfhub.dev/google/yamnet/1), then compressing the model with
TensorFlow Lite quantization so it can run on low-power edge hardware (Raspberry Pi, Arduino Nano 33 BLE).

Data and project context: Sunbird AI noise-pollution project. The research proposal is in
`Masters Proposal Defense_ Tibabwetiza Muhanguzi.pdf`.

## Repository contents

| File | Purpose |
|---|---|
| `urbannoiseclassifierv4.ipynb` | Main notebook: data loading, embedding extraction, training, evaluation, export, TFLite quantization, Edge Impulse profiling/deployment. Opens in Colab via the badge at the top of the notebook. |
| `3Aug2023-9_tflite_model.tflite` | Final quantized (uint8 in/out) classifier head, about 784 KB. |
| `record-infer-upload.py` | Edge device script: records audio, classifies it, writes a results file, uploads it to the metrics dashboard. |
| `cmergedv2.py` | Simpler edge test script: records 10 s clips in a loop and prints the predicted class. |
| `Masters Proposal Defense_ ...pdf` | Proposal defence slides. |

## Approach

1. **Embeddings.** Audio is resampled to 16 kHz mono and passed through the frozen YAMNet model, producing a
   1024-dimension embedding per ~0.96 s frame.
2. **Classifier head.** A small dense network is trained on the embeddings:
   `Dense(512, relu) -> Dropout(0.5) -> Dense(512, relu) -> Dropout(0.5) -> Dense(19)`
   (about 797k parameters, 3.04 MB in float32). Adam optimizer, sparse categorical cross-entropy, up to 50 epochs
   with early stopping.
3. **Serving model.** YAMNet + classifier + a mean over frames are saved as one SavedModel that takes a raw
   waveform and returns one 19-way score vector per clip.
4. **Quantization.** The classifier head is converted to TFLite for edge deployment (see below).

### Classes (19)

`car-or-truck`, `motorvehicle-horn`, `boda-boda-motocyle`, `motorvehicle-siren`, `car-alarm`, `mobile-music`,
`hawker-vendor`, `community-radio`, `regilious-venue`, `herbalists`, `construction-site`, `fabrication-workshop`,
`generator`, `bar-restaurant-nightclub`, `animal`, `crowd-noise`, `school`, `street-preacher`, `other`

(Label strings are kept exactly as used in the code and data folders, including the `regilious-venue` spelling.)

### Dataset

- 52 labelled WAV clips per class, 988 clips total, perfectly balanced across the 19 classes.
- Split 80 / 10 / 10 into train / validation / test at the embedding-frame level.
- The audio is not included in this repository (it lives on the project's shared Google Drive).

## Results

| Metric | Value |
|---|---|
| Test accuracy (float32 head) | **87.2%** |
| Test loss | 0.489 |
| Final train accuracy | about 82% (dropout active during training) |
| Final validation accuracy | about 88% |

A confusion matrix and training curves are rendered in the notebook (cells for `plot_history` and
`confusion_matrix`).

Note that the split is made on embedding frames, so frames from the same recording can land in both train and test.
The 87.2% figure is therefore likely optimistic; a split by recording would give a stricter estimate.

### Sanity check on a field recording

A real recording (`SB1005-2023-06-12T211309-81_converted.wav`) was classified as `motorvehicle-siren` with
probability 0.505, identically by the in-memory model and the reloaded SavedModel.

## Quantization and deployment

The notebook documents several conversion attempts, including the ones that failed:

| Attempt | Settings | Outcome |
|---|---|---|
| `test1.tflite` | `Optimize.DEFAULT` + float16 supported types | Converted successfully. |
| `test2.tflite` | `Optimize.DEFAULT` + representative dataset (100 samples) | Converted successfully. |
| Full-integer | INT8 builtins, uint8 input/output, representative data, applied to the **full waveform model** (YAMNet included) | **Failed**: `PAD` op error (`paddings` dims 1 != 3). YAMNet's front end does not quantize cleanly. |
| Final model | Quantize only the **classifier head** on 1024-d embeddings | Works. Input `[1, 1024]` uint8, output `[1, 19]` uint8. |
| Edge Impulse | `ei.model.profile` / `ei.model.deploy` with `AudioInput(16000)` | Used to estimate RAM/ROM/latency across targets and to build an Arduino C++ library. |
| TinyML C array | `tinymlgen.port` and a manual `hex_to_c_array` | Generates `model_code.h` / `updatedlib.h` for microcontroller use. |

Key finding: quantizing the whole YAMNet pipeline failed, so the deployed design splits the work. Feature
extraction (YAMNet) runs separately and the small quantized head does the classification. The final TFLite file is
about 784 KB (about 802,960 bytes), roughly a 4x reduction from the 3.04 MB float32 head. Edge Impulse reported the
model as supported on the Arduino Nano 33 BLE profile and on MPU-class devices.

## Deployment variants

The same classifier was taken to three kinds of target. Only the Raspberry Pi path has runnable code in this
repository; the microcontroller path is documented from the notebook (see the note below the table).

| Target | Model format | Where YAMNet runs | How it was produced | In this repo |
|---|---|---|---|---|
| Raspberry Pi 4 (MPU) | Keras SavedModel (`embedmodel/`), full waveform to 19 scores | On device, in TensorFlow | `serving_model.save(...)` in the notebook | `record-infer-upload.py`, `cmergedv2.py` |
| Any MPU / mobile | TFLite (float16 or dynamic-range) | On device | `test1.tflite`, `test2.tflite` cells | notebook only |
| Microcontroller (Arduino Nano 33 BLE and similar) | INT8 TFLite head, `[1,1024]` uint8 in, `[1,19]` uint8 out | Must be computed separately and passed in as the 1024-d embedding | Edge Impulse `ei.model.deploy(..., deploy_target='arduino')`, plus `tinymlgen` / `hex_to_c_array` C headers | `3Aug2023-9_tflite_model.tflite` (model only) |

### Microcontroller profile (Edge Impulse estimates, INT8 head)

| Device class | Inference time | RAM (TFLite / EON) | ROM (TFLite / EON) |
|---|---|---|---|
| Arduino Nano 33 BLE | 38 ms | 5.3 KB / 3.3 KB | 827 KB / 812 KB |
| Cortex-M0+ at 40 MHz | 343 ms | 5.2 KB / 3.3 KB | 825 KB / 811 KB |
| Cortex-M7 at 240 MHz | 6 ms | 5.3 KB / 3.3 KB | 827 KB / 812 KB |
| Cortex-A72 class MPU | 1 ms | n/a | 803 KB |

The model file is 802,960 bytes. These are profiler estimates, not measurements on hardware.

**Important caveats**

- The profile covers the **classifier head only**. YAMNet's feature extraction (MobileNet-style, several MB) is not
  included, and it is what failed full-integer quantization. A standalone microcontroller still needs a way to produce
  the 1024-d embedding, for example an on-device YAMNet port or the embedding computed on a gateway/phone and sent
  over BLE.
- The head alone uses about 812 to 827 KB of ROM, which is most of the Nano 33 BLE's 1 MB flash, so it fits but leaves
  little room for the application.
- The Arduino library zip (`blenoiselib.zip`) and the generated C headers (`model_code.h`, `updatedlib.h`) were
  written to Google Drive / Colab storage and are **not** committed here. The notebook cells that generate them are
  the source of truth.

### Generating the microcontroller artifacts

From `urbannoiseclassifierv4.ipynb`:

1. **Edge Impulse Arduino library:** set `ei.API_KEY` from an environment variable, then run the `ei.model.deploy`
   cell with `AudioInput(16000)` and `Classification(labels=my_classes)`. Unzip the result and add it through
   Arduino IDE: Sketch > Include Library > Add .ZIP Library.
2. **Raw C array for TensorFlow Lite Micro:** run the `hex_to_c_array` cell on the quantized model bytes to produce a
   header you can include in a TFLM sketch.
3. **tinymlgen:** `port(my_model, optimize=True)` writes `model_code.h` for the Keras head directly.

## Running the Raspberry Pi scripts

Tested design: Raspberry Pi with a USB microphone (`arecord -D plughw:3,0`).

```bash
pip install tensorflow tensorflow-io numpy requests
# SavedModel directory expected at ./embedmodel
python record-infer-upload.py
```

`record-infer-upload.py` records 6 clips, classifies each, saves the clip as
`inference/<n>-<class>-<prob>.wav`, writes per-class probabilities to `inference_results.txt`, and POSTs that file to
the dashboard endpoint with the device ID. The hourly `while True` loop and 10-minute sleep are currently commented
out for testing. Create the `inference/` directory before running.

## Reproducing the training

1. Open `urbannoiseclassifierv4.ipynb` in Colab.
2. Point `load_data(...)` at a folder containing one sub-folder of `.wav` files per class name.
3. Run all cells down to the `serving_model.save(...)` cell, then the TFLite cells.

Paths in the notebook point to a private Google Drive and will need changing.

## Known issues and caveats

- **Credential exposed.** The notebook contains a hard-coded Edge Impulse API key (cell `ei.API_KEY = ...`). Revoke
  that key in Edge Impulse and load a replacement from an environment variable instead.
- Some cell outputs are stale (for example `pd_data.shape` reports 988 rows after the dataset variable was
  reassigned to the 190-row balanced subset), so re-run the notebook before trusting printed shapes.
- The INT8 accuracy of the final `.tflite` is not reported in the notebook. The 87.2% above is for the float32 head.
  Evaluate the quantized model on the test set to confirm the accuracy cost of quantization.
- Early stopping monitors training `loss`, not `val_loss`.

## Author

Tibabwetiza Joel Muhanguzi, Masters research.
