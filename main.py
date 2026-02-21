import argparse
import os
import tkinter as tk

from cnn_model import ChessPieceClassifier
from engine import StockfishUCI
from gui import ChessAssistantApp
from vision import ChessVision


def parse_args():
    p = argparse.ArgumentParser(description="CNN Chess Assistant with Stockfish GUI")
    p.add_argument("--model-path", default="models/chess_cnn.pt", help="Path to trained CNN checkpoint.")
    p.add_argument("--stockfish-path", default="stockfish-windows-x86-64-avx2.exe", help="Path to Stockfish.")
    p.add_argument("--device", default="cpu", help="Torch device: cpu or cuda.")
    p.add_argument("--side-to-move", default="w", choices=["w", "b"], help="Side to move in generated FEN.")
    p.add_argument("--update-interval", type=float, default=1.2, help="Vision update interval (seconds).")
    p.add_argument("--engine-interval", type=float, default=2.0, help="Engine refresh interval (seconds).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.stockfish_path):
        raise FileNotFoundError(f"Stockfish not found: {args.stockfish_path}")

    classifier = ChessPieceClassifier(checkpoint_path=args.model_path, device=args.device)
    vision = ChessVision(classifier=classifier, side_to_move=args.side_to_move)
    engine = StockfishUCI(engine_path=args.stockfish_path)

    root = tk.Tk()
    _app = ChessAssistantApp(
        root=root,
        vision=vision,
        engine=engine,
        update_interval_sec=args.update_interval,
        engine_interval_sec=args.engine_interval,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
