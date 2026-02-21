# CNN Chess Assistant Project

This project uses a **13-class CNN** (`12 pieces + empty`) to read a live chessboard from screen, build FEN, and query Stockfish.

## Files

- `prepare_dataset.py` -> build augmented dataset from your 26 seed images in `grayscale_images/`.
- `train_cnn.py` -> train CNN and save checkpoint.
- `cnn_model.py` -> model + preprocessing + batched inference.
- `vision.py` -> board selection, 8x8 split, CNN inference, FEN generation.
- `engine.py` -> Stockfish UCI wrapper.
- `gui.py` -> live GUI with board view, eval bar, best move, and controls.
- `main.py` -> run app.

## 1) Prepare dataset from your 26 images

Your current input format is correct:

- `grayscale_images/white_pawn/*.png`
- ...
- `grayscale_images/black_king/*.png`
- `grayscale_images/empty/*.png`

Generate augmented dataset:

```bash
python prepare_dataset.py --source-root grayscale_images --output-root dataset --target-per-class 116 --image-size 64
```

`116 * 13 = 1508` total images (about your 1500 target).

## 2) Train CNN

```bash
python train_cnn.py --data-root dataset --epochs 18 --batch-size 128 --device cpu --save-path models/chess_cnn.pt
```

The checkpoint saves:

- model weights
- class names
- image size
- mean/std normalization

So inference preprocessing always matches training.

## 3) Run live assistant GUI

```bash
python main.py --model-path models/chess_cnn.pt --stockfish-path stockfish-windows-x86-64-avx2.exe --device cpu
```

## GUI controls

- `Select Board` -> drag-select board area on screen.
- `Pause/Resume` -> pause live updates.
- `Evaluate Now` -> force immediate Stockfish analysis.

## Required packages

```bash
pip install torch opencv-python mss numpy
```

