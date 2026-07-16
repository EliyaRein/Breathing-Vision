"""CPU latency benchmark for the deployment pipeline.

Runs the REAL StreamingPipeline (incremental Lucas-Kanade optical flow +
per-window IQR norm + GRU inference) on a real mp4, forced onto CPU, and reports:
  * optical-flow cost per frame (frames that only track, no scoring)
  * full per-frame cost on frames that also score a window
  * pure model inference cost per scored window (isolated, N repeats)
  * real-time headroom vs the 100 ms/frame budget at 10 fps
ROI = full frame (a conservative upper bound: larger crop => larger LK pyramid).
"""
import sys, time
import numpy as np
import cv2
import torch

torch.set_num_threads(max(1, torch.get_num_threads()))  # default CPU threads
from deploy.pipeline import StreamingPipeline

TARGET_FPS = 10.0
MAX_FRAMES = 400


def resampled_frames(path):
    # Fractional resample to TARGET_FPS, matching dataset_builder/app _select_indices
    # (frame k -> round(k*ofps/TARGET_FPS)); integer striding would give 7.5 fps on
    # a 15 fps source.
    cap = cv2.VideoCapture(path)
    ofps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    frames, i, k, last = [], 0, 0, -1
    while len(frames) < MAX_FRAMES:
        ok, fr = cap.read()
        if not ok:
            break
        target = int(round(k * ofps / TARGET_FPS))
        if i == target and i > last:
            frames.append(fr)
            last = i
            k += 1
        i += 1
    cap.release()
    return frames, ofps


def bench_clip(path):
    frames, ofps = resampled_frames(path)
    if len(frames) < 90:
        return None
    h, w = frames[0].shape[:2]
    pipe = StreamingPipeline(box=(0, 0, w, h), device="cpu")

    of_only, of_score = [], []
    for fr in frames:
        n_before = len(pipe.breaths)
        t0 = time.perf_counter()
        scored_t0 = pipe._next_t0                 # noqa: SLF001 (introspection)
        pipe.push(fr)
        dt = (time.perf_counter() - t0) * 1e3     # ms
        # did this frame trigger at least one window score?
        if pipe._next_t0 != scored_t0:
            of_score.append(dt)
        else:
            of_only.append(dt)

    # isolate pure inference on a full buffer
    inf = []
    t0 = pipe.n - pipe.wcfg.test - pipe.wcfg.future - 1
    t0 = max(pipe.wcfg.past, t0)
    for _ in range(200):
        s = time.perf_counter()
        pipe._score_window(t0)                    # noqa: SLF001
        inf.append((time.perf_counter() - s) * 1e3)

    return dict(path=path, ofps=ofps, res=f"{w}x{h}",
                n=len(frames),
                of_only=np.array(of_only), of_score=np.array(of_score),
                inf=np.array(inf))


def stat(a):
    return f"{a.mean():.2f}±{a.std():.2f} ms  (median {np.median(a):.2f}, p95 {np.percentile(a,95):.2f})"


def main():
    clips = sys.argv[1:]
    if not clips:
        print("usage: python -m tools.bench_cpu <video.mp4> [more.mp4 ...]")
        sys.exit(1)
    print(f"torch CPU threads = {torch.get_num_threads()}")
    for path in clips:
        try:
            r = bench_clip(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if r is None:
            print(f"skip {path}: too short")
            continue
        print(f"\n=== {r['path']}")
        print(f"  orig_fps={r['ofps']:.1f}  resampled->{TARGET_FPS:.0f}fps  "
              f"res={r['res']}  frames={r['n']}")
        print(f"  optical-flow / frame (track only):   {stat(r['of_only'])}")
        print(f"  frame that also scores a window:     {stat(r['of_score'])}")
        print(f"  pure GRU inference / window:          {stat(r['inf'])}")
        budget = 1000.0 / TARGET_FPS
        # worst case = the heaviest frames (those that also score a window), not
        # diluted by the many track-only frames.
        worst = np.percentile(r['of_score'], 95) if len(r['of_score']) else 0.0
        print(f"  budget @10fps = {budget:.0f} ms/frame  ->  worst-frame p95 "
              f"= {worst:.2f} ms  ->  ~{budget/max(worst,1e-6):.0f}x real time")


if __name__ == "__main__":
    main()
