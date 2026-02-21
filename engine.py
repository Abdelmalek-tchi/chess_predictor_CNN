import subprocess
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class EngineOutput:
    best_move: str
    eval_text: str
    eval_cp_white: float


class StockfishUCI:
    def __init__(self, engine_path: str, threads: int = 2, hash_mb: int = 128) -> None:
        self.engine_path = engine_path
        self.threads = int(threads)
        self.hash_mb = int(hash_mb)
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.last_output = EngineOutput(best_move="-", eval_text="-", eval_cp_white=0.0)

    def _is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, force_restart: bool = False) -> None:
        if force_restart:
            self.stop()
        if self._is_running():
            return
        self.proc = None
        self.proc = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("isready")
        self._read_until("readyok")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        self.proc = None

    def _send(self, cmd: str) -> None:
        if not self._is_running() or self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Stockfish process not running.")
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(f"Failed sending UCI command '{cmd}': {exc}") from exc

    def _read_until(self, token: str) -> None:
        if not self._is_running() or self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Stockfish process not running.")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish output closed unexpectedly.")
            if token in line:
                return

    @staticmethod
    def _score_to_white_cp(score_line: str, stm: str) -> float:
        if " score cp " in score_line:
            cp = float(score_line.split(" score cp ", 1)[1].split()[0])
            return cp if stm == "w" else -cp
        if " score mate " in score_line:
            mate = int(score_line.split(" score mate ", 1)[1].split()[0])
            cp = 100000.0 if mate > 0 else -100000.0
            return cp if stm == "w" else -cp
        return 0.0

    @staticmethod
    def _score_to_text(score_line: str, stm: str) -> str:
        if " score cp " in score_line:
            cp = int(float(score_line.split(" score cp ", 1)[1].split()[0]))
            cp = cp if stm == "w" else -cp
            return f"{cp:+d} cp"
        if " score mate " in score_line:
            mate = int(score_line.split(" score mate ", 1)[1].split()[0])
            mate = mate if stm == "w" else -mate
            return f"M{mate:+d}"
        return "-"

    def analyze(self, fen: str, movetime_ms: int = 200) -> EngineOutput:
        with self.lock:
            last_exc: Optional[Exception] = None
            for attempt in range(2):
                try:
                    self.start(force_restart=(attempt > 0))
                    stm = fen.split()[1] if len(fen.split()) > 1 else "w"
                    self._send("isready")
                    self._read_until("readyok")
                    self._send(f"position fen {fen}")
                    self._send(f"go movetime {int(movetime_ms)}")

                    info_score = ""
                    best_move = "-"
                    if self.proc is None or self.proc.stdout is None:
                        raise RuntimeError("Stockfish process not running.")
                    while True:
                        line = self.proc.stdout.readline()
                        if not line:
                            raise RuntimeError("Stockfish output ended unexpectedly.")
                        line = line.strip()
                        if line.startswith("info") and " score " in line:
                            info_score = line
                        if line.startswith("bestmove"):
                            parts = line.split()
                            if len(parts) >= 2:
                                best_move = parts[1]
                            break

                    out = EngineOutput(
                        best_move=best_move,
                        eval_text=self._score_to_text(info_score, stm),
                        eval_cp_white=self._score_to_white_cp(info_score, stm),
                    )
                    self.last_output = out
                    return out
                except Exception as exc:
                    last_exc = exc
                    self.stop()

            if last_exc is not None:
                return EngineOutput(
                    best_move=self.last_output.best_move,
                    eval_text=f"{self.last_output.eval_text} (recovered)",
                    eval_cp_white=self.last_output.eval_cp_white,
                )
            return self.last_output
