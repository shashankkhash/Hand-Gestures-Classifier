# Down-Gesture Detector page

A one-page explainer of this project: the dataset, the network, the training curve and the mistakes.

- `gesture-detector.html` – the built page (open it in a browser).
- `train_numpy.py` – NumPy port of `Nueral_network.py` (same 960-100-1 sigmoid network, init, per-sample SGD, eta 0.1, 1000 epochs, seed 7). Writes `results.json`. Takes about a minute.
- `build_page.py` – embeds `results.json` and sample images into `template.html` to produce the page.

The image lists name 21 files that are not in `gestures/`; both scripts skip them, so training uses the 246 photos that exist.

```
pip install numpy pillow
python showcase/train_numpy.py
python showcase/build_page.py
```

Published version: https://claude.ai/artifact/AbBtmTWCJW8fadsEbqsJvP
