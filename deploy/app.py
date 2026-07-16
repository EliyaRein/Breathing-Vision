"""Breathing-Vision -- streaming breath-detection desktop app.

Flow:
  1. open a video, seek to the first non-black frame;
  2. pick the ROI -- auto (YOLO person box), manual drag, or the whole frame;
  3. stream the video resampled to 10 fps through the frozen GRU pipeline;
  4. show the frame with a breath-LED flash + live BPM/breaths cards;
  5. on end, show a summary (duration, breaths, mean rate, longest gap).

The on-screen video is delayed by the model's decode latency so the breath-LED
flash lands on (roughly) the exact frame the breath occurred in.

The heavy work (optical flow + inference) runs in a QThread so the UI stays
responsive; the thread emits an annotated frame + stats per 10-fps step.

Run:  python -m deploy.app
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# The torch/ultralytics-backed modules are imported lazily in _load_backend() so
# the splash screen can paint before the multi-second torch load (see main()).
BuildConfig = _select_indices = find_first_real_frame = detect_roi = None
StreamingPipeline = DEFAULT_CKPT = None

DISPLAY_W = 900          # max on-screen width for frames
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
SYNC_OFFSET = 1          # extra display-delay frames on top of the decode latency
                         # (empirically 15 frames feels perfectly synced)
FLASH_LIFE = 3           # frames the breath LED stays lit after a detection
APNEA_SEC = 10.0         # no detected breath for this long -> apnea alert.
                         # NOTE: heuristic, UNVALIDATED -- we have no apnea-labelled
                         # data to tune/verify it.


# --------------------------------------------------------------------------- #
# overlay drawing
# --------------------------------------------------------------------------- #
def draw_overlay(frame, box, apnea_gap=0.0, flash=0.0):
    """Annotate a BGR frame: ROI box, a top-right breath LED (lights up on a fresh
    detection, `flash` in 0..1), and (if `apnea_gap` exceeds APNEA_SEC) a red
    no-breath alert. Breath count / rate live in the styled Qt cards."""
    img = frame.copy()
    x1, y1, x2, y2 = box
    alert = apnea_gap >= APNEA_SEC
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 80), 2)

    # breath LED (top-right corner): dim ring when idle, bright red pulse on a breath
    h, w = img.shape[:2]
    lx, ly = w - 34, 34
    if flash > 0:
        r = int(11 + 7 * flash)
        glow = img.copy()
        cv2.circle(glow, (lx, ly), r, (60, 60, 255), -1)     # BGR bright red
        cv2.addWeighted(glow, min(1.0, 0.35 + 0.65 * flash), img,
                        1 - min(1.0, 0.35 + 0.65 * flash), 0, img)
        cv2.circle(img, (lx, ly), r, (255, 255, 255), 2)
    else:
        cv2.circle(img, (lx, ly), 11, (70, 70, 95), 2)       # dim idle ring

    if alert:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        msg = f"! NO BREATH {apnea_gap:.0f}s"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        px = (w - tw) // 2
        cv2.rectangle(img, (px - 12, 48), (px + tw + 12, 92), (0, 0, 200), -1)
        cv2.putText(img, msg, (px, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 3)
    return img


# --------------------------------------------------------------------------- #
# streaming worker
# --------------------------------------------------------------------------- #
class StreamWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object, float, int, float)
    progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, video_path, box, first_idx, ckpt=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.box = tuple(int(v) for v in box)
        self.first_idx = int(first_idx)
        self.ckpt = ckpt or DEFAULT_CKPT
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            pipe = StreamingPipeline(self.box, self.ckpt)
        except FileNotFoundError:
            self.error.emit(f"Model checkpoint not found:\n{self.ckpt}\n\n"
                            "Train it first:  python -m deploy.train_deploy")
            return
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Failed to load model:\n{e}")
            return

        cfg = BuildConfig()
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        targets = set(_select_indices(self.first_idx, n_frames, fps, cfg))
        if not targets:
            self.error.emit("No frames to process after 10-fps resampling.")
            cap.release()
            return

        # Display is delayed by the pipeline's decode latency so the breath-LED
        # flash lands on the SAME frame the breath actually occurred in (the
        # detection is only known ~latency frames later). We buffer the last
        # `latency+1` frames and always show the oldest, drawing the flash on it.
        from collections import deque
        L = pipe.latency + SYNC_OFFSET
        buf: deque = deque(maxlen=L + 1)            # (pushed_idx, frame)
        stats = {"max_gap": 0.0, "alerts": 0, "in_apnea": False, "last": -1}

        def render(d, dframe):
            """Draw + emit the frame at pushed-index d, synced to its breaths."""
            n_b = sum(1 for b in pipe.breaths if b <= d)
            prior = [b for b in pipe.breaths if b <= d]
            gap = (d - (prior[-1] if prior else 0)) / pipe.fps
            stats["max_gap"] = max(stats["max_gap"], gap)
            if gap >= APNEA_SEC and not stats["in_apnea"]:
                stats["alerts"] += 1
                stats["in_apnea"] = True
            elif gap < APNEA_SEC:
                stats["in_apnea"] = False
            bpm = pipe.bpm_upto(d)
            ages = [d - b for b in pipe.breaths if 0 <= d - b <= FLASH_LIFE]
            flash = 1.0 - (min(ages) / FLASH_LIFE) if ages else 0.0
            disp = draw_overlay(dframe, self.box, gap, flash)
            self.frame_ready.emit(disp, bpm, n_b, gap)
            stats["last"] = d

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_idx)
        cur = self.first_idx
        while not self._stop and cur < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if cur in targets:
                pipe.push(frame)
                p = pipe.n - 1                       # index just pushed
                buf.append((p, frame))
                if len(buf) == L + 1:                # oldest is exactly frame p-L
                    d, dframe = buf[0]
                    render(d, dframe)
                    self.msleep(70)                  # keep playback watchable
                self.progress.emit(cur, n_frames)
            cur += 1
        cap.release()
        # flush the tail (frames still held back by the display delay); covers the
        # short-clip case too, where the buffer never filled and nothing showed yet
        for d, dframe in list(buf):
            if self._stop:
                break
            if d > stats["last"]:
                render(d, dframe)
                self.msleep(70)

        summary = pipe.summary()
        summary["max_gap_s"] = stats["max_gap"]
        summary["apnea_alerts"] = stats["alerts"]
        self.done.emit(summary)


# --------------------------------------------------------------------------- #
# ROI selection dialog (auto / manual / full-frame)
# --------------------------------------------------------------------------- #
class RoiDialog(QtWidgets.QDialog):
    def __init__(self, frame, video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select region of interest")
        self.setModal(True)
        self.frame = frame
        self.video_path = video_path
        self.h, self.w = frame.shape[:2]
        self.scale = min(1.0, DISPLAY_W / self.w)
        self.box = None                 # (x1,y1,x2,y2) full-res
        self._drag_start = None
        self._drag_cur = None

        self.label = QtWidgets.QLabel()
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self._press
        self.label.mouseMoveEvent = self._move
        self.label.mouseReleaseEvent = self._release

        self.info = QtWidgets.QLabel("Drag on the image to draw a box, or use the "
                                     "buttons below.")
        auto_btn = QtWidgets.QPushButton("Auto-detect (YOLO)")
        full_btn = QtWidgets.QPushButton("Use full frame")
        self.ok_btn = QtWidgets.QPushButton("Start")
        self.ok_btn.setObjectName("primary")
        self.ok_btn.setEnabled(False)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        auto_btn.clicked.connect(self._auto)
        full_btn.clicked.connect(self._full)
        self.ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        btns = QtWidgets.QHBoxLayout()
        for b in (auto_btn, full_btn, self.ok_btn, cancel_btn):
            btns.addWidget(b)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(self.info)
        lay.addLayout(btns)
        self._redraw()

    # --- coordinate mapping (display <-> full-res) --------------------------
    def _to_full(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _redraw(self):
        img = self.frame.copy()
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 80), 2)
        elif self._drag_start and self._drag_cur:
            # draw the live preview in full-res coords (the image is scaled down
            # afterwards), so it stays synced with the cursor like the final box.
            cv2.rectangle(img, self._to_full(*self._drag_start),
                          self._to_full(*self._drag_cur), (60, 160, 255), 2)
        disp = cv2.resize(img, (int(self.w * self.scale), int(self.h * self.scale)))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(disp.data, disp.shape[1], disp.shape[0],
                            disp.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.label.setFixedSize(disp.shape[1], disp.shape[0])

    # --- mouse (manual box) -------------------------------------------------
    def _press(self, e):
        self._drag_start = (e.pos().x(), e.pos().y())
        self._drag_cur = self._drag_start
        self.box = None

    def _move(self, e):
        if self._drag_start:
            self._drag_cur = (e.pos().x(), e.pos().y())
            self._redraw()

    def _release(self, e):
        if not self._drag_start:
            return
        (x1, y1), (x2, y2) = self._drag_start, (e.pos().x(), e.pos().y())
        self._drag_start = self._drag_cur = None
        fx1, fy1 = self._to_full(min(x1, x2), min(y1, y2))
        fx2, fy2 = self._to_full(max(x1, x2), max(y1, y2))
        if fx2 - fx1 < 96 or fy2 - fy1 < 96:
            self.info.setText("Box too small -- drag a larger region (min 96x96 px).")
            self.box = None
            self._redraw()
            return
        self._set_box((fx1, fy1, fx2, fy2), "manual")

    # --- auto / full --------------------------------------------------------
    def _auto(self):
        try:
            from ultralytics import YOLO
        except Exception:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "YOLO unavailable",
                                          "ultralytics is not installed.")
            return
        cfg = BuildConfig()
        self.info.setText("Running YOLO detection...")
        QtWidgets.QApplication.processEvents()
        try:
            model = YOLO(cfg.yolo_weights)
            roi = detect_roi(self.video_path, model, cfg)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Detection failed", str(e))
            return
        src = roi.source + (f" (conf {roi.confidence:.2f})" if roi.source == "yolo" else "")
        self._set_box(tuple(int(v) for v in roi.box), src)

    def _full(self):
        self._set_box((0, 0, self.w, self.h), "full frame")

    def _set_box(self, box, src):
        self.box = box
        self.ok_btn.setEnabled(True)
        self.info.setText(f"ROI = {box}  [{src}]")
        self._redraw()


# --------------------------------------------------------------------------- #
# main window
# --------------------------------------------------------------------------- #
def _stat_card(title):
    """A small dark 'card' with a caption and a big value label (returned)."""
    card = QtWidgets.QFrame()
    card.setObjectName("card")
    v = QtWidgets.QVBoxLayout(card)
    v.setContentsMargins(16, 10, 16, 12)
    v.setSpacing(2)
    cap = QtWidgets.QLabel(title)
    cap.setObjectName("caption")
    val = QtWidgets.QLabel("--")
    val.setObjectName("value")
    v.addWidget(cap)
    v.addWidget(val)
    return card, val


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breathing-Vision")
        if os.path.exists(LOGO_PATH):
            self.setWindowIcon(QtGui.QIcon(LOGO_PATH))
        self.worker = None

        # header: logo + title/subtitle
        logo = QtWidgets.QLabel()
        if os.path.exists(LOGO_PATH):
            logo.setPixmap(QtGui.QPixmap(LOGO_PATH).scaled(
                46, 46, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        title = QtWidgets.QLabel("Breathing-Vision")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("camera-based infant breath detection")
        subtitle.setObjectName("subtitle")
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(logo)
        head.addLayout(titles)

        # buttons
        self.open_btn = QtWidgets.QPushButton("Open video")
        self.open_btn.setObjectName("primary")
        self.open_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_video)
        self.stop_btn.clicked.connect(self.stop_stream)

        top = QtWidgets.QHBoxLayout()
        top.addLayout(head)
        top.addStretch(1)
        top.addWidget(self.open_btn)
        top.addWidget(self.stop_btn)

        # stat cards
        bpm_card, self.bpm_value = _stat_card("RATE (BPM)")
        breath_card, self.breath_value = _stat_card("BREATHS")
        status_card, self.status_value = _stat_card("STATUS")
        self.status_value.setText("idle")
        cards = QtWidgets.QHBoxLayout()
        cards.setSpacing(12)
        for c in (bpm_card, breath_card, status_card):
            cards.addWidget(c, stretch=1)

        # video area
        self.video_label = QtWidgets.QLabel("Open a video to begin")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 460)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(18, 16, 18, 12)
        lay.setSpacing(14)
        lay.addLayout(top)
        lay.addLayout(cards)
        lay.addWidget(self.video_label, stretch=1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    # -- open + ROI ----------------------------------------------------------
    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "", "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Error", "Could not open video.")
            return
        first_idx, frame = find_first_real_frame(cap, BuildConfig())
        cap.release()
        if frame is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Could not read any frame.")
            return

        dlg = RoiDialog(frame, path, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted or dlg.box is None:
            return
        self.start_stream(path, dlg.box, first_idx)

    # -- stream --------------------------------------------------------------
    def start_stream(self, path, box, first_idx):
        self.worker = StreamWorker(path, box, first_idx)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.open_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.bpm_value.setText("--")
        self.breath_value.setText("0")
        self.status_value.setText("streaming")
        self.statusBar().showMessage("Streaming...")
        self.worker.start()

    def stop_stream(self):
        if self.worker:
            self.worker.stop()

    # -- signal handlers -----------------------------------------------------
    def on_frame(self, img_bgr, bpm, n_breaths, gap):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.bpm_value.setText(f"{bpm:.1f}" if bpm > 0 else "--")
        self.breath_value.setText(str(n_breaths))
        if gap >= APNEA_SEC:
            self.status_value.setText(f"APNEA  {gap:.0f}s")
            self.status_value.setProperty("state", "alert")
        else:
            self.status_value.setText("breathing")
            self.status_value.setProperty("state", "ok")
        # re-apply style so the dynamic 'state' property takes effect
        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)

    def on_progress(self, cur, total):
        if total:
            self.statusBar().showMessage(f"Streaming... {100*cur//total}%")

    def on_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self._reset_buttons()

    def on_done(self, summary):
        self._reset_buttons()
        self.status_value.setText("done")
        self.status_value.setProperty("state", "")
        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)
        self.statusBar().showMessage("Done")
        gap = summary.get("max_gap_s", 0.0)
        alerts = summary.get("apnea_alerts", 0)
        apnea_line = (f"\nLongest no-breath gap: {gap:.1f} s"
                      f"   (apnea alerts: {alerts})")
        QtWidgets.QMessageBox.information(
            self, "Summary",
            f"Duration: {summary['duration_s']:.1f} s "
            f"({summary['n_frames']} frames @10fps)\n"
            f"Breaths detected: {summary['n_breaths']}\n"
            f"Mean rate: {summary['bpm']:.1f} BPM"
            f"{apnea_line}\n\n"
            "Detections may include missed or false breaths. "
            "Not for medical decisions. Video is not stored.")

    def _reset_buttons(self):
        self.open_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        e.accept()


STYLE = """
* { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
QMainWindow, QWidget { background: #0e1116; color: #e6e9ef; }
QLabel#title { font-size: 22px; font-weight: 700; color: #f2f4f8; }
QLabel#subtitle { font-size: 12px; color: #7d8798; }
QFrame#card {
    background: #171b22; border: 1px solid #232a34; border-radius: 12px;
}
QLabel#caption { font-size: 11px; font-weight: 600; color: #7d8798;
    letter-spacing: 1px; }
QLabel#value { font-size: 30px; font-weight: 700; color: #eaf2ff; }
QLabel#value[state="ok"]    { color: #4ade80; }
QLabel#value[state="alert"] { color: #ff5c5c; }
QLabel#video {
    background: #05070a; border: 1px solid #232a34; border-radius: 14px;
    color: #5b6675; font-size: 15px;
}
QPushButton {
    background: #1b212b; color: #e6e9ef; border: 1px solid #2b3341;
    border-radius: 9px; padding: 9px 18px; font-size: 13px; font-weight: 600;
}
QPushButton:hover { background: #222a36; border-color: #3a4557; }
QPushButton:disabled { color: #566072; background: #141922; border-color: #1f2632; }
QPushButton#primary { background: #2f6df6; border: none; color: #ffffff; }
QPushButton#primary:hover { background: #4880ff; }
QPushButton#primary:disabled { background: #24304a; color: #7f8ba4; }
QStatusBar { background: #0b0e13; color: #7d8798; }
QStatusBar::item { border: none; }
"""


def make_splash():
    """Compose the startup splash pixmap: logo + wordmark + copyright."""
    W, H = 640, 420
    pix = QtGui.QPixmap(W, H)
    pix.fill(QtGui.QColor("#0e1116"))
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    p.setPen(QtGui.QColor("#232a34"))
    p.drawRoundedRect(1, 1, W - 2, H - 2, 16, 16)

    if os.path.exists(LOGO_PATH):
        logo = QtGui.QPixmap(LOGO_PATH).scaled(
            150, 150, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        p.drawPixmap((W - logo.width()) // 2, 56, logo)

    word = QtGui.QFont("Segoe UI", 28, QtGui.QFont.Bold)
    word.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 3)
    p.setFont(word)
    p.setPen(QtGui.QColor("#eaf2ff"))
    p.drawText(QtCore.QRect(0, 222, W, 64),
               QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, "BREATHING-VISION")

    p.setFont(QtGui.QFont("Segoe UI", 11))
    p.setPen(QtGui.QColor("#7d8798"))
    p.drawText(QtCore.QRect(0, 292, W, 24), QtCore.Qt.AlignHCenter,
               "camera-based infant breath detection")

    p.setFont(QtGui.QFont("Segoe UI", 8))
    p.setPen(QtGui.QColor("#566072"))
    p.drawText(QtCore.QRect(0, 324, W, 18), QtCore.Qt.AlignHCenter,
               "Not a medical device \u00B7 research / educational use only")

    p.setFont(QtGui.QFont("Segoe UI", 9))
    p.setPen(QtGui.QColor("#566072"))
    p.drawText(QtCore.QRect(0, H - 40, W, 20), QtCore.Qt.AlignHCenter,
               "\u00A9 2026 Elia Reinstein \u00B7 Shira Barmats")
    p.end()
    return pix


def _load_backend():
    """Import the torch/ultralytics-backed modules and bind them as globals.

    Kept out of module import so the splash can paint before the multi-second
    torch load; called once from main() while the splash is on screen.
    """
    global BuildConfig, _select_indices, find_first_real_frame, detect_roi
    global StreamingPipeline, DEFAULT_CKPT
    from dataset_builder.config import BuildConfig
    from dataset_builder.motion import _select_indices
    from dataset_builder.roi import find_first_real_frame, detect_roi
    from deploy.pipeline import StreamingPipeline, DEFAULT_CKPT


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    if os.path.exists(LOGO_PATH):
        app.setWindowIcon(QtGui.QIcon(LOGO_PATH))

    splash = QtWidgets.QSplashScreen(make_splash(), QtCore.Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("  Loading models\u2026",
                       QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
                       QtGui.QColor("#7d8798"))
    app.processEvents()                  # paint the splash before the heavy import

    _load_backend()                      # multi-second torch/ultralytics load

    win = MainWindow()
    win.resize(1040, 760)
    win.show()
    splash.finish(win)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
