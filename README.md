# Video Surveillance — Detection, Tracking & Event Recognition

A modular, CLI-driven computer vision pipeline that detects people in video, tracks them across frames with persistent IDs, and raises configurable zone-based events (intrusion, loitering). Built around **Pipeline**, **Factory**, **Strategy**, and **Observer** design patterns so each stage can be modified, replaced, or tested in isolation.

```
python run.py --video input.mp4 --zones zones.json --output results/
```

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Choices](#model-choices)
3. [Setup Instructions](#setup-instructions)
4. [Configuration](#configuration)
5. [Sample Results](#sample-results)
6. [Known Limitations](#known-limitations)
7. [Performance Notes](#performance-notes)

---

## Architecture Overview

The pipeline is a linear sequence of stages, each with a single responsibility. Frames move through the pipeline one at a time so memory use stays bounded regardless of video length.

```
┌────────────┐     ┌─────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  Video I/O │ --> │  Detection  │ --> │  Tracking  │ --> │   Event    │ --> │   Output   │
│ (frames)   │     │  (YOLOv8)   │     │ (DeepSORT/ │     │ Detection  │     │ (video +   │
│            │     │             │     │ ByteTrack) │     │ (zones)    │     │  logs)     │
└────────────┘     └─────────────┘     └────────────┘     └────────────┘     └────────────┘
      │                    │                  │                  │                  │
 utils/video_io.py   detectors/          trackers/           events/            utils/logger.py
                     (Factory)           (Strategy)          (Observer)
```

**Stage responsibilities**

| Stage | Module | Role |
|---|---|---|
| Video I/O | `utils/video_io.py` | Reads frames (with optional stride/resize), writes the annotated output video, handles corrupt/empty frames |
| Detection | `detectors/` | Runs YOLOv8 on each frame, returns bounding boxes + confidence scores for the "person" class |
| Tracking | `trackers/` | Assigns persistent IDs across frames, re-identifies people who leave and re-enter the frame |
| Event Detection | `events/` | Checks track positions against zone polygons, raises `zone_intrusion` and `loitering` events |
| Output / Logging | `utils/logger.py` | Writes `events.csv` / `events.json` and `tracks.csv`; draws overlays onto the output video |

**Design patterns and why each one fits**

- **Pipeline Pattern** — the five stages above are chained as a sequence of independent, swappable steps. A stage only needs to know the shape of the data it receives and produces, not how upstream/downstream stages work internally.
- **Factory Pattern** — `detectors/detector_factory.py` builds the configured detector (e.g. YOLOv8n/s/m) from a config string, so adding a new detector doesn't require touching the pipeline or CLI.
- **Strategy Pattern** — trackers implement a common interface (`update(detections) -> tracks`), so DeepSORT and ByteTrack are interchangeable at runtime via `--tracker`.
- **Observer Pattern** — `EventLogger` (in `utils/logger.py`) subscribes to the event detector and is notified via `update(event)` whenever a zone event fires. This decouples "detecting an event" from "persisting an event," so additional observers (e.g. a live dashboard, an alert webhook) can be added without changing event-detection logic.

---

## Model Choices

| Component | Chosen | Why | Alternatives considered |
|---|---|---|---|
| Object detector | **YOLOv8n** (Ultralytics) | Best speed/accuracy trade-off for CPU/edge-friendly inference; small model size; active maintenance and simple Python API; free/open license | YOLOv8s/m (higher accuracy, slower — used when GPU is available), Faster R-CNN (too slow for near-real-time use), SSD-MobileNet (faster but noticeably lower accuracy on small/occluded people) |
| Tracker (default) | **DeepSORT** | Appearance embeddings help re-identify people after occlusion or brief exits from frame, which the assignment explicitly requires | ByteTrack (faster, motion-only — offered as `--tracker bytetrack` for high-FPS/low-occlusion scenes), SORT (no appearance model, more ID switches) |
| Tracker (alternative) | **ByteTrack** | Associates low-confidence detections instead of discarding them, which helps in crowded scenes; no embedding network, so it's lighter and faster | Kept as a Strategy option rather than the default because it re-identifies less reliably after a full occlusion |
| Zone geometry | **Shapely polygons** | Robust point-in-polygon and overlap checks for arbitrary (non-rectangular) zones defined in `zones.json` | Manual bounding-box overlap math (rejected: doesn't support arbitrary polygons) |

**Rule of thumb used in this project:** DeepSORT + YOLOv8n is the default because re-identification accuracy matters more than raw speed for surveillance; ByteTrack is offered for throughput-sensitive deployments where brief ID switches are acceptable.

---

## Setup Instructions

### Requirements
- Python 3.9+
- pip
- (Optional) CUDA-capable GPU for faster inference

### Install

```bash
git clone https://github.com/MadhushreeRamachandran/video-surveillance.git
cd video-surveillance
uv sync
cd src
```

`requirements.txt` includes: `ultralytics`, `deep-sort-realtime`, `opencv-python`, `shapely`, `numpy`.

### Run

```bash
uv run python run.py --video ..\data\clips\lighting_test.mp4 --zones ..\data\zones\lighting_test.zones.json --output ..\outputs\lighting_test --tracker bytetrack --conf 0.15 --frame-skip 1

```

### CLI options

| Flag | Required | Default | Description |
|---|---|---|---|
| `--video` | Yes | — | Path to input video file |
| `--zones` | Yes | — | Path to zones config JSON |
| `--output` | Yes | — | Output directory for video + logs |
| `--detector` | No | `yolov8n` | Detector model name/weights |
| `--tracker` | No | `deepsort` | `deepsort` or `bytetrack` |
| `--conf` | No | `0.4` | Detection confidence threshold |
| `--stride` | No | `1` | Process every Nth frame (speed/accuracy trade-off) |
| `--device` | No | auto | `cpu`, `cuda`, or `cuda:0` |
| `--log-level` | No | `INFO` | `DEBUG`, `INFO`, `WARNING` |

### Outputs (written to `--output`)

```
results/
├── annotated.mp4     # video with boxes, IDs, zone overlays, event banners
├── events.csv        # streamed event log (crash-safe)
├── events.json        # final event log + run summary
└── tracks.csv         # per-frame bounding boxes + confidence (MOT-format friendly)
```

---

## Configuration

### `zones.json`

Zones are arbitrary polygons in pixel coordinates, each with its own event rules.

```json
{
  "zones": [
    {
      "name": "entrance",
      "polygon": [[100, 200], [400, 200], [400, 500], [100, 500]],
      "events": {
        "intrusion": true,
        "loitering": { "enabled": true, "threshold_sec": 10 }
      }
    },
    {
      "name": "restricted_area",
      "polygon": [[600, 100], [900, 100], [900, 400], [600, 400]],
      "events": {
        "intrusion": true,
        "loitering": { "enabled": false }
      }
    }
  ]
}
```

- `polygon`: list of `[x, y]` pixel coordinates (any number of vertices, not just rectangles).
- `intrusion`: fires once when a track first enters the zone.
- `loitering.threshold_sec`: fires once a track has remained inside the zone continuously for this many seconds.

### Adjustable thresholds (CLI or config)

| Parameter | Where | Purpose |
|---|---|---|
| `--conf` | CLI | Minimum detection confidence to keep a box |
| `loitering.threshold_sec` | `zones.json` | Dwell time before loitering fires |
| `dedup_window_frames` | `EventLogger` init (in `run.py`) | Suppresses repeat events for the same track/zone within N frames |
| `--stride` | CLI | Trade FPS for lower compute by skipping frames |
| IoU / max-age (tracker) | tracker config | Occlusion tolerance before a track ID is dropped |

---

## Sample Results



- **Annotated video:** `results/annotated.mp4` — bounding boxes with track IDs, zone polygons drawn as overlays, on-screen event banners when intrusion/loitering fires.
- **Event log excerpt (`events.json`):**

```json
{
  "summary": {
    "total_events": 4,
    "events_by_type": { "zone_intrusion": 3, "loitering": 1 },
    "unique_tracks_with_events": 3
  },
  "events": [
    {
      "event_id": 1,
      "event_type": "zone_intrusion",
      "track_id": 7,
      "zone_name": "entrance",
      "frame_number": 142,
      "timestamp_sec": 5.68,
      "bbox": [412.3, 210.5, 480.1, 390.2],
      "confidence": 0.87
    }
  ]
}
```

---

## Known Limitations

- **Low light:** detection confidence drops noticeably in poorly lit scenes; YOLOv8n was not fine-tuned on low-light data, so recall suffers more than precision.
- **Crowded / heavy occlusion:** dense crowds increase ID switches even with DeepSORT's appearance model; long full occlusions can still cause a track to be dropped and reassigned a new ID on reappearance.
- **Re-identification window:** re-identification only works within DeepSORT's embedding gallery lifetime — a person absent for a very long time (well beyond `max_age`) will get a new ID.
- **Camera motion:** the pipeline assumes a mostly static camera; pans/zooms are not compensated for and can produce spurious loitering/intrusion events.
- **No multi-camera handoff:** each video is processed independently; there's no cross-camera identity matching.
- **With more time, I would add:** a Kalman-filter-based motion compensation step for camera shake, a lightweight MOTA/MOTP evaluation script against MOT17 ground truth, alert deduplication across overlapping zones, and a live dashboard (stretch goals from the assignment).

---

## Performance Notes

> Placeholder — fill in with numbers from your own hardware/test clips before submission.

| Setup | Detector | Tracker | Resolution | FPS (approx.) | Peak memory |
|---|---|---|---|---|---|
| CPU (example) | YOLOv8n | DeepSORT | 640×480 | *TBD* | *TBD* |
| GPU (example) | YOLOv8n | DeepSORT | 640×480 | *TBD* | *TBD* |
| GPU (example) | YOLOv8n | ByteTrack | 640×480 | *TBD* | *TBD* |

- **GPU/CPU awareness:** `--device` selects CPU or CUDA; falls back to CPU automatically if CUDA is unavailable.
- **Memory handling:** frames are streamed one at a time (`utils/video_io.py`), so memory use does not grow with video length; only the tracker's embedding gallery scales with the number of concurrently tracked people.
- **Throughput tuning:** `--stride` skips frames to trade detail for speed; ByteTrack is lighter than DeepSORT since it skips the appearance-embedding network.
- **Benchmarking:** run with `--log-level DEBUG` to see per-frame timing breakdowns (detection / tracking / event-check / write) in the console log.