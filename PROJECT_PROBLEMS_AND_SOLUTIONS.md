# Chess Live Predictor: Problems and Solutions

## 1. Problem: Stockfish works for a while, then stops updating
**Solution:**
- Added robust engine error handling in `StockfishAnalyzer.analyze()`.
- If FEN is invalid, the app now keeps the last valid engine output instead of returning empty values.
- If engine process errors/terminates, the app now attempts automatic restart.
- Added analysis throttling (`engine_interval`) to avoid overloading continuous engine calls.

## 2. Problem: No visibility into engine health/status
**Solution:**
- Added live engine status tracking (`self.status`) in `StockfishAnalyzer`.
- Added GUI label `Engine: ...` to show states such as `running`, `invalid FEN (using last)`, `restarted`, and error fallback states.

## 3. Problem: Detected piece colors can be wrong and flicker frame-to-frame
**Solution:**
- Kept temporal smoothing (`_stabilize_matrix`) to reduce noise across recent frames.
- Kept color continuity logic (`_fix_color_swaps_with_counts`) to prevent sudden unrealistic color flips.
- Added warning detection for suspicious opposite-color edge-rank placements (`_opposite_color_warning`).

## 4. Problem: User could not manually correct wrong piece color on board
**Solution:**
- Added board click interaction (`<Button-1>` on canvas).
- Clicking a non-empty square now toggles that piece color (white <-> black).
- Added persistent per-square manual overrides so user correction is preserved across incoming frames.

## 5. Problem: Manual edits could be overwritten by next vision update
**Solution:**
- Added `_apply_manual_color_overrides()` in live processing path.
- Manual color overrides are applied to each new matrix before FEN conversion and rendering.
- Overrides are removed automatically if square becomes empty.

## 6. Problem: Vision pipeline can produce invalid positions for engine analysis
**Solution:**
- Analyzer now gracefully handles invalid FEN parse (`ValueError`) and continues using last valid engine result.
- This prevents UI breakdown and keeps best move/eval stable during brief detection noise.

## 7. Problem: Hard to diagnose if issue is vision or engine
**Solution:**
- UI now exposes confidence, piece counts, color warning, and engine status together.
- This makes it easier to tell whether failures come from board detection or Stockfish runtime.

## 8. Problem: Real-time loop can become unstable under high update pressure
**Solution:**
- Introduced controlled engine call cadence (`~8 calls/sec` by default with `engine_interval=0.12`).
- Keeps capture loop responsive while avoiding excessive engine churn.

## Recommended Next Improvements
**Problem:** No persisted logs for post-run debugging.
**Solution:** Add lightweight logging (timestamp, FEN, engine status, exceptions) to a local `.log` file.

**Problem:** No confidence threshold before sending positions to engine.
**Solution:** Skip engine calls when confidence is below a configurable threshold and keep last stable result.

**Problem:** Manual color corrections are not currently saved between app sessions.
**Solution:** Save/load overrides in a small JSON file tied to board region.
