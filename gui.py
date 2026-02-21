import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox
from typing import Dict, List, Optional, Tuple

from engine import EngineOutput, StockfishUCI
from vision import BoardRegion, BoardSelector, ChessVision, VisionFrame


@dataclass
class UIState:
    vision: Optional[VisionFrame] = None
    engine_by_side: Dict[str, EngineOutput] = field(default_factory=dict)
    running: bool = True
    paused: bool = False
    last_fen_piece: str = ""
    force_eval: bool = False


class ChessAssistantApp:
    def __init__(
        self,
        root: tk.Tk,
        vision: ChessVision,
        engine: StockfishUCI,
        update_interval_sec: float = 1.2,
        engine_interval_sec: float = 2.0,
    ) -> None:
        self.root = root
        self.vision = vision
        self.engine = engine
        self.update_interval_sec = float(update_interval_sec)
        self.engine_interval_sec = float(engine_interval_sec)
        self.board_region: Optional[BoardRegion] = None
        self.last_engine_ts = 0.0

        self.state = UIState()
        self.lock = threading.Lock()

        self.auto_side_to_move = self.vision.side_to_move if self.vision.side_to_move in ("w", "b") else "w"
        self.side_override = "auto"  # auto | w | b
        self.suggest_mode = "both"  # w | b | both
        self.castling_rights: Dict[str, bool] = {"K": False, "Q": False, "k": False, "q": False}
        self.en_passant = "-"

        self.current_fen_full = "8/8/8/8/8/8/8/8 w - - 0 1"
        self.current_arrows: List[Tuple[str, str]] = []

        self.status_var = tk.StringVar(value="No board selected")
        self.region_var = tk.StringVar(value="Region: -")
        self.fen_var = tk.StringVar(value="FEN: -")
        self.best_var = tk.StringVar(value="Suggested move(s): -")
        self.eval_var = tk.StringVar(value="Eval: -")
        self.conf_var = tk.StringVar(value="Confidence: -")
        self.detected_side_var = tk.StringVar(value="Detected side: white")
        self.used_side_var = tk.StringVar(value="Using side: white")
        self.white_move_var = tk.StringVar(value="White best: -")
        self.black_move_var = tk.StringVar(value="Black best: -")

        self.side_mode_var = tk.StringVar(value="auto")
        self.suggest_mode_var = tk.StringVar(value="both")
        self.castle_k_var = tk.BooleanVar(value=False)
        self.castle_q_var = tk.BooleanVar(value=False)
        self.castle_k_black_var = tk.BooleanVar(value=False)
        self.castle_q_black_var = tk.BooleanVar(value=False)
        self.ep_choice_var = tk.StringVar(value="-")

        self._build_ui()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.root.after(100, self._refresh)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.title("Chess Assistant CNN + Stockfish")
        self.root.geometry("1260x780")

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)
        tk.Button(top, text="Select Board", width=16, command=self._select_board).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Pause/Resume", width=16, command=self._toggle_pause).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Evaluate Now", width=16, command=self._evaluate_now).pack(side=tk.LEFT, padx=4)
        tk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        body = tk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.board_canvas = tk.Canvas(left, width=700, height=700, bg="#111111", highlightthickness=0)
        self.board_canvas.pack(anchor="w")
        self._draw_board(self.current_fen_full, [])

        right = tk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=14)

        tk.Label(right, textvariable=self.region_var, anchor="w", justify="left").pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.detected_side_var, anchor="w", justify="left").pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.used_side_var, anchor="w", justify="left").pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.fen_var, anchor="w", justify="left", wraplength=500).pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.best_var, anchor="w", justify="left", wraplength=500).pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.white_move_var, anchor="w", justify="left").pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.black_move_var, anchor="w", justify="left").pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.eval_var, anchor="w", justify="left").pack(fill=tk.X, pady=2)
        tk.Label(right, textvariable=self.conf_var, anchor="w", justify="left").pack(fill=tk.X, pady=2)

        side_frame = tk.LabelFrame(right, text="Side To Move")
        side_frame.pack(fill=tk.X, pady=(10, 6))
        tk.Radiobutton(
            side_frame,
            text="Auto detect",
            variable=self.side_mode_var,
            value="auto",
            command=lambda: self._set_side_override("auto"),
        ).pack(anchor="w")
        tk.Radiobutton(
            side_frame,
            text="Force white to move",
            variable=self.side_mode_var,
            value="w",
            command=lambda: self._set_side_override("w"),
        ).pack(anchor="w")
        tk.Radiobutton(
            side_frame,
            text="Force black to move",
            variable=self.side_mode_var,
            value="b",
            command=lambda: self._set_side_override("b"),
        ).pack(anchor="w")

        suggest_frame = tk.LabelFrame(right, text="Suggest Moves For")
        suggest_frame.pack(fill=tk.X, pady=6)
        tk.Radiobutton(
            suggest_frame,
            text="White",
            variable=self.suggest_mode_var,
            value="w",
            command=lambda: self._set_suggest_mode("w"),
        ).pack(anchor="w")
        tk.Radiobutton(
            suggest_frame,
            text="Black",
            variable=self.suggest_mode_var,
            value="b",
            command=lambda: self._set_suggest_mode("b"),
        ).pack(anchor="w")
        tk.Radiobutton(
            suggest_frame,
            text="Both",
            variable=self.suggest_mode_var,
            value="both",
            command=lambda: self._set_suggest_mode("both"),
        ).pack(anchor="w")

        castle_frame = tk.LabelFrame(right, text="Castling Rights")
        castle_frame.pack(fill=tk.X, pady=6)
        row1 = tk.Frame(castle_frame)
        row1.pack(fill=tk.X)
        tk.Checkbutton(row1, text="K", variable=self.castle_k_var, command=self._sync_castling_from_vars).pack(
            side=tk.LEFT, padx=4
        )
        tk.Checkbutton(row1, text="Q", variable=self.castle_q_var, command=self._sync_castling_from_vars).pack(
            side=tk.LEFT, padx=4
        )
        tk.Checkbutton(
            row1, text="k", variable=self.castle_k_black_var, command=self._sync_castling_from_vars
        ).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            row1, text="q", variable=self.castle_q_black_var, command=self._sync_castling_from_vars
        ).pack(side=tk.LEFT, padx=4)

        row2 = tk.Frame(castle_frame)
        row2.pack(fill=tk.X, pady=(4, 2))
        tk.Button(row2, text="Start KQkq", command=self._set_castling_start).pack(side=tk.LEFT, padx=4)
        tk.Button(row2, text="No Castling", command=self._set_castling_none).pack(side=tk.LEFT, padx=4)

        ep_frame = tk.LabelFrame(right, text="En Passant")
        ep_frame.pack(fill=tk.X, pady=6)
        ep_options = ["-"] + [f"{f}3" for f in "abcdefgh"] + [f"{f}6" for f in "abcdefgh"]
        tk.OptionMenu(ep_frame, self.ep_choice_var, *ep_options).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(ep_frame, text="Apply EP", command=self._apply_ep_choice).pack(side=tk.LEFT, padx=4)
        tk.Button(ep_frame, text="Clear EP", command=self._clear_ep).pack(side=tk.LEFT, padx=4)

        tk.Label(right, text="Evaluation Bar (White POV)").pack(anchor="w", pady=(12, 6))
        self.eval_canvas = tk.Canvas(right, width=80, height=280, bg="#1a1a1a", highlightthickness=0)
        self.eval_canvas.pack(anchor="w")
        self._draw_eval_bar(0.0)

    def _set_side_override(self, mode: str) -> None:
        if mode not in ("auto", "w", "b"):
            return
        with self.lock:
            self.side_override = mode
            self.state.force_eval = True

    def _set_suggest_mode(self, mode: str) -> None:
        if mode not in ("w", "b", "both"):
            return
        with self.lock:
            self.suggest_mode = mode
            self.state.force_eval = True

    def _sync_castling_from_vars(self) -> None:
        with self.lock:
            self.castling_rights["K"] = bool(self.castle_k_var.get())
            self.castling_rights["Q"] = bool(self.castle_q_var.get())
            self.castling_rights["k"] = bool(self.castle_k_black_var.get())
            self.castling_rights["q"] = bool(self.castle_q_black_var.get())
            self.state.force_eval = True

    def _set_castling_start(self) -> None:
        self.castle_k_var.set(True)
        self.castle_q_var.set(True)
        self.castle_k_black_var.set(True)
        self.castle_q_black_var.set(True)
        self._sync_castling_from_vars()

    def _set_castling_none(self) -> None:
        self.castle_k_var.set(False)
        self.castle_q_var.set(False)
        self.castle_k_black_var.set(False)
        self.castle_q_black_var.set(False)
        self._sync_castling_from_vars()

    @staticmethod
    def _normalize_ep(value: str) -> str:
        if value is None:
            return "-"
        s = value.strip().lower()
        if s == "-":
            return "-"
        if len(s) == 2 and s[0] in "abcdefgh" and s[1] in "36":
            return s
        return "-"

    def _apply_ep_choice(self) -> None:
        with self.lock:
            self.en_passant = self._normalize_ep(self.ep_choice_var.get())
            self.state.force_eval = True

    def _clear_ep(self) -> None:
        self.ep_choice_var.set("-")
        with self.lock:
            self.en_passant = "-"
            self.state.force_eval = True

    def _castling_field(self) -> str:
        rights = "".join([c for c in "KQkq" if self.castling_rights.get(c, False)])
        return rights if rights else "-"

    def _effective_side_to_move(self) -> str:
        if self.side_override == "w":
            return "w"
        if self.side_override == "b":
            return "b"
        return self.auto_side_to_move

    def _build_full_fen(self, fen_piece: str, side_to_move: str, castling: str, ep: str) -> str:
        return f"{fen_piece} {side_to_move} {castling} {ep} 0 1"

    def _select_board(self) -> None:
        selector = BoardSelector(self.root)
        selected = selector.select()
        if not selected:
            self.status_var.set("Selection canceled")
            return
        self.board_region = selected
        self.region_var.set(
            f"Region: left={selected.left}, top={selected.top}, width={selected.width}, height={selected.height}"
        )
        self.status_var.set("Running")

    def _toggle_pause(self) -> None:
        with self.lock:
            self.state.paused = not self.state.paused
            paused = self.state.paused
        self.status_var.set("Paused" if paused else "Running")

    def _evaluate_now(self) -> None:
        with self.lock:
            self.state.force_eval = True
        self.status_var.set("Manual evaluate requested")

    def _loop(self) -> None:
        while True:
            with self.lock:
                if not self.state.running:
                    break
                paused = self.state.paused
            if paused or self.board_region is None:
                time.sleep(0.1)
                continue

            try:
                vf = self.vision.capture_and_predict(self.board_region)
                now = time.time()

                with self.lock:
                    prev_piece = self.state.last_fen_piece
                    force = self.state.force_eval
                    self.state.force_eval = False

                    if prev_piece and vf.fen_piece != prev_piece:
                        self.auto_side_to_move = "b" if self.auto_side_to_move == "w" else "w"

                    self.state.last_fen_piece = vf.fen_piece
                    self.state.vision = vf

                    side_for_eval = self._effective_side_to_move()
                    suggest_mode = self.suggest_mode
                    castling = self._castling_field()
                    ep = self.en_passant

                should_eval = False
                if vf.fen_piece != prev_piece:
                    should_eval = True
                if now - self.last_engine_ts >= self.engine_interval_sec:
                    should_eval = True
                if force:
                    should_eval = True

                if should_eval:
                    results: Dict[str, EngineOutput] = {}

                    if suggest_mode in ("w", "both"):
                        fen_w = self._build_full_fen(vf.fen_piece, "w", castling, ep)
                        results["w"] = self.engine.analyze(fen_w, movetime_ms=220)

                    if suggest_mode in ("b", "both"):
                        fen_b = self._build_full_fen(vf.fen_piece, "b", castling, ep)
                        results["b"] = self.engine.analyze(fen_b, movetime_ms=220)

                    if side_for_eval not in results:
                        fen_eval = self._build_full_fen(vf.fen_piece, side_for_eval, castling, ep)
                        results[side_for_eval] = self.engine.analyze(fen_eval, movetime_ms=220)

                    with self.lock:
                        self.state.engine_by_side = results
                    self.last_engine_ts = now

                time.sleep(self.update_interval_sec)
            except Exception as exc:
                with self.lock:
                    self.state.paused = True
                self.root.after(0, lambda e=exc: messagebox.showerror("Runtime error", str(e)))
                time.sleep(0.2)

    @staticmethod
    def _side_label(side: str) -> str:
        return "white" if side == "w" else "black"

    def _refresh(self) -> None:
        with self.lock:
            state = self.state
            vf = state.vision
            engine_by_side = dict(state.engine_by_side)
            paused = state.paused
            auto_side = self.auto_side_to_move
            side_override = self.side_override
            suggest_mode = self.suggest_mode
            castling = self._castling_field()
            ep = self.en_passant

        if self.board_region is None:
            self.status_var.set("No board selected")
        elif paused:
            self.status_var.set("Paused")
        else:
            self.status_var.set("Running")

        used_side = auto_side if side_override == "auto" else side_override
        self.detected_side_var.set(f"Detected side: {self._side_label(auto_side)}")
        self.used_side_var.set(f"Using side: {self._side_label(used_side)}")

        if vf is not None:
            self.current_fen_full = self._build_full_fen(vf.fen_piece, used_side, castling, ep)
            self.fen_var.set(f"FEN: {self.current_fen_full}")
            suffix = " (flipped)" if vf.flipped else ""
            self.conf_var.set(f"Confidence: {vf.confidence_avg:.3f}{suffix}")

        white_out = engine_by_side.get("w")
        black_out = engine_by_side.get("b")
        self.white_move_var.set(f"White best: {white_out.best_move if white_out else '-'}")
        self.black_move_var.set(f"Black best: {black_out.best_move if black_out else '-'}")

        arrows: List[Tuple[str, str]] = []
        if suggest_mode == "w" and white_out:
            self.best_var.set(f"Suggested move(s): White {white_out.best_move}")
            arrows.append((white_out.best_move, "#3aa7ff"))
        elif suggest_mode == "b" and black_out:
            self.best_var.set(f"Suggested move(s): Black {black_out.best_move}")
            arrows.append((black_out.best_move, "#ff5a36"))
        else:
            wm = white_out.best_move if white_out else "-"
            bm = black_out.best_move if black_out else "-"
            self.best_var.set(f"Suggested move(s): White {wm} | Black {bm}")
            if white_out:
                arrows.append((white_out.best_move, "#3aa7ff"))
            if black_out:
                arrows.append((black_out.best_move, "#ff5a36"))

        eval_out = engine_by_side.get(used_side) or white_out or black_out
        if eval_out is not None:
            self.eval_var.set(f"Eval: {eval_out.eval_text}")
            self._draw_eval_bar(eval_out.eval_cp_white)

        self.current_arrows = arrows
        self._draw_board(self.current_fen_full, self.current_arrows)
        self.root.after(120, self._refresh)

    @staticmethod
    def _parse_uci_move(best_move: str):
        if best_move is None:
            return None
        mv = best_move.strip().lower()
        if len(mv) < 4:
            return None
        src = mv[:2]
        dst = mv[2:4]
        valid = "abcdefgh"
        if src[0] not in valid or dst[0] not in valid:
            return None
        if src[1] not in "12345678" or dst[1] not in "12345678":
            return None
        return src, dst

    @staticmethod
    def _square_center(square: str, origin_x: int, origin_y: int, cell: float):
        file_idx = ord(square[0]) - ord("a")
        rank_idx = int(square[1]) - 1
        x = origin_x + (file_idx + 0.5) * cell
        y = origin_y + (7 - rank_idx + 0.5) * cell
        return x, y

    @staticmethod
    def _fen_board_matrix(fen_full: str) -> List[List[str]]:
        piece_part = fen_full.split()[0] if fen_full else "8/8/8/8/8/8/8/8"
        rows = piece_part.split("/")
        if len(rows) != 8:
            rows = ["8"] * 8
        out: List[List[str]] = []
        for row in rows:
            arr: List[str] = []
            for ch in row:
                if ch.isdigit():
                    arr.extend(["."] * int(ch))
                else:
                    arr.append(ch)
            if len(arr) < 8:
                arr.extend(["."] * (8 - len(arr)))
            out.append(arr[:8])
        return out

    def _draw_board(self, fen_full: str, arrows: List[Tuple[str, str]]) -> None:
        c = self.board_canvas
        c.delete("all")

        board_size = 640
        ox = 30
        oy = 30
        cell = board_size / 8.0
        light = "#f0d9b5"
        dark = "#b58863"

        for r in range(8):
            for col in range(8):
                x1 = ox + col * cell
                y1 = oy + r * cell
                x2 = x1 + cell
                y2 = y1 + cell
                fill = light if (r + col) % 2 == 0 else dark
                c.create_rectangle(x1, y1, x2, y2, fill=fill, outline=fill)

        board = self._fen_board_matrix(fen_full)
        for r in range(8):
            for col in range(8):
                p = board[r][col]
                if p == ".":
                    continue
                x = ox + (col + 0.5) * cell
                y = oy + (r + 0.5) * cell
                is_white = p.isupper()
                bg = "#f8f8f8" if is_white else "#121212"
                fg = "#111111" if is_white else "#f2f2f2"
                c.create_oval(x - 20, y - 20, x + 20, y + 20, fill=bg, outline="#333333", width=1)
                c.create_text(x, y, text=p.upper(), fill=fg, font=("Consolas", 16, "bold"))

        for i, file_char in enumerate("abcdefgh"):
            x = ox + (i + 0.5) * cell
            c.create_text(x, oy + board_size + 16, text=file_char, fill="#d8d8d8", font=("Consolas", 10))
        for i in range(8):
            y = oy + (i + 0.5) * cell
            c.create_text(ox - 12, y, text=str(8 - i), fill="#d8d8d8", font=("Consolas", 10))

        for move, color in arrows:
            parsed = self._parse_uci_move(move)
            if parsed is None:
                continue
            src, dst = parsed
            x1, y1 = self._square_center(src, ox, oy, cell)
            x2, y2 = self._square_center(dst, ox, oy, cell)
            c.create_line(x1, y1, x2, y2, fill=color, width=7, arrow=tk.LAST, arrowshape=(20, 24, 8))
            c.create_oval(x1 - 7, y1 - 7, x1 + 7, y1 + 7, fill="#0e0e0e", outline="#f8f8f8", width=2)

        c.create_rectangle(ox, oy, ox + board_size, oy + board_size, outline="#111111", width=3)

    def _draw_eval_bar(self, cp_white: float) -> None:
        cp = max(-1200.0, min(1200.0, float(cp_white)))
        frac_white = 0.5 + cp / 2400.0
        frac_white = max(0.02, min(0.98, frac_white))
        h = 280
        w = 80
        white_h = int(h * frac_white)
        black_h = h - white_h

        c = self.eval_canvas
        c.delete("all")
        c.create_rectangle(0, 0, w, black_h, fill="#121212", outline="#2a2a2a")
        c.create_rectangle(0, black_h, w, h, fill="#efefef", outline="#2a2a2a")
        c.create_line(0, black_h, w, black_h, fill="#ff8c00", width=2)
        c.create_text(40, 14, text="Black", fill="#dedede", font=("Consolas", 10, "bold"))
        c.create_text(40, h - 14, text="White", fill="#202020", font=("Consolas", 10, "bold"))

    def _on_close(self) -> None:
        with self.lock:
            self.state.running = False
        try:
            self.engine.stop()
        except Exception:
            pass
        self.root.destroy()
