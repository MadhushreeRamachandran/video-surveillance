# Video Surveillance: Detection, Tracking & Event Recognition

A CLI tool that takes a video, detects and tracks people in it, and raises alerts when someone enters a restricted zone or lingers too long in one. Built for the AI/ML intern take-home assignment.

```
python run.py --video input.mp4 --zones zones.json --output results/
```

Output: an annotated video with boxes/IDs/zone overlays, plus JSON and CSV event logs.

## Architecture

The pipeline is a straight line of stages. Each frame flows through all of them in order, and each stage only knows about the one before it.

```
video file
   |
   v
[ VideoReader ]        reads frames one at a time, keeps real timestamps
   |
   v
[ Detection Stage ]    YOLOv8 finds people in the frame
   |
   v
[ Tracking Stage ]     DeepSORT/ByteTrack assigns/keeps IDs across frames
   |
   v
[ Event Stage ]        checks each tracked person against the zone polygons
   |
   v
[ Annotation Stage ]   draws boxes, IDs, zones, alerts on the frame
   |
   v
[ VideoWriter ] + [ JSON/CSV loggers ]
```

I used four patterns to keep this modular:

- **Pipeline** – the stages above are separate classes (`DetectionStage`, `TrackingStage`, `EventStage`, `AnnotationStage`) run in sequence by `SurveillancePipeline`. Adding a new stage means writing one class, not touching the others.
- **Factory** – `create_detector("yolov8")` and `create_tracker("deepsort")` build objects from a name. Swapping models is a config change.
- **Strategy** – DeepSORT and ByteTrack both implement the same `Tracker` interface, so the pipeline doesn't care which one is running.
- **Observer** – event logging (console, JSON, CSV) is done by observers that subscribe to an `EventPublisher`. The zone logic doesn't know or care who's listening.

Frames are streamed one at a time through the whole thing — nothing loads the full video into memory, so runtime scales with video length, not with how much RAM you have.

## Model choices

**Detector: YOLOv8n (nano)**

I picked the nano variant because this is a real-time streaming use case, not a batch analysis job. YOLOv8n runs usably fast even on CPU, at the cost of some accuracy on small/distant people. I considered Faster R-CNN, which is generally more accurate on small objects, but it's a two-stage detector and noticeably slower — a bad fit if this ever needs to run on more than a handful of cameras. If accuracy on small/far-away people becomes a real problem, the fix is swapping to `yolov8s.pt` or `yolov8m.pt` — same code, different weights file, since the detector is behind a Factory.

**Tracker: DeepSORT (default), ByteTrack (alternative)**

The assignment specifically asks for re-identification — a person leaving and re-entering the frame should keep the same ID. Only an appearance-based tracker can do that, which is why DeepSORT is the default: it uses a small CNN to embed each detected person and match them by appearance, not just position.

The cost is speed — that embedding step runs per person per frame, and it's the main reason this pipeline is slow on CPU. ByteTrack is the alternative: pure motion-based tracking, no appearance model, much faster, but a person who's out of frame for more than a couple of seconds will get a new ID when they come back. I'd use ByteTrack over DeepSORT for crowded scenes on limited hardware where speed and occlusion-robustness matter more than long-term re-identification.

Both are selectable with `--tracker deepsort` or `--tracker bytetrack`.

## Setup

Requires Python 3.9–3.12.

**With uv (what I used):**

```bash
uv sync
uv run python run.py --video input.mp4 --zones zones.json --output results/
```

**With plain pip:**

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python run.py --video input.mp4 --zones zones.json --output results/
```

YOLOv8n weights download automatically on first run. No GPU needed — it auto-detects CUDA and falls back to CPU.

## Configuration

**Zones** are polygons defined in a JSON file, in normalized (0–1) coordinates so the same file works regardless of resolution:

```json
{
  "coordinate_mode": "normalized",
  "defaults": {
    "loiter_seconds": 10,
    "min_inside_seconds": 0.3,
    "exit_grace_seconds": 2.0,
    "cooldown_seconds": 10
  },
  "zones": [
    {
      "id": "restricted_area",
      "name": "Restricted Area",
      "polygon": [[0.1, 0.45], [0.45, 0.45], [0.5, 0.9], [0.1, 0.9]],
      "events": ["intrusion", "loitering"]
    }
  ]
}
```

- `loiter_seconds` — how long someone has to stay roughly still inside a zone before it counts as loitering.
- `min_inside_seconds` — how long someone has to be inside a zone before an intrusion fires (filters out boundary flicker).
- `exit_grace_seconds` — how long someone can be briefly out of view before their zone visit is considered over (handles short occlusions).
- `cooldown_seconds` — stops the same person triggering the same event repeatedly.

Any of these can be overridden per zone. Coordinates use the person's foot position (bottom-center of the box), which lines up with where they're actually standing better than the box center does.

**Main CLI flags:**

```
--video          input video path
--zones          zones JSON path
--output         output folder
--tracker        deepsort (default) or bytetrack
--conf           detector confidence threshold
--frame-skip     process every Nth frame, for speed
--device         auto / cpu / cuda:0
--no-video       skip writing the annotated video, events only
```

## Sample results

Ran on a ~75 second indoor clip (480x360, gym/hall setting with a registration table and a seating area, both marked as zones):

```json
{
  "summary": { "total_events": 20, "by_type": { "zone_intrusion": 10, "loitering": 10 } },
  "events": [
    {
      "event_type": "zone_intrusion",
      "zone_name": "Registration Desk",
      "track_id": 4,
      "frame": 87,
      "timecode": "00:00:02.900",
      "confidence": 0.891
    },
    {
      "event_type": "loitering",
      "zone_name": "Seating Area",
      "track_id": 7,
      "frame": 412,
      "timecode": "00:00:13.730",
      "confidence": 0.763,
      "details": { "stationary_seconds": 15.02 }
    }
  ]
}
```

The seating-area loitering events are people who were genuinely just sitting in chairs — which is correct behavior for the algorithm. It has no concept of intent, only "did this person stay roughly still, in this zone, past the threshold." Whether that's actually suspicious depends entirely on how zones and thresholds are set up for a real deployment.

The annotated video draws each person's box in a per-ID color, switches to red/orange with a thicker outline when they trigger an alert, and shows a small HUD (frame number, timecode, FPS, active track count) in the corner.

## Edge cases and how they're handled

- **Empty or corrupt frames** — `VideoReader` skips unreadable frames instead of crashing, and only stops if it hits 30 in a row (real end of file). The detector also returns an empty list for a `None` frame instead of erroring.
- **Occlusion (person briefly blocked from view)** — DeepSORT keeps a track alive for `max_age` frames after it stops seeing that person, and matches them back by appearance if they reappear. On the test clip, I checked a person passing near others in the seating area and their ID held across the brief overlap.
- **ID switches** — this is a real limitation, not something I could fully solve. If someone is out of frame for longer than DeepSORT's re-identification window (default ~90 frames, about 3 seconds), they come back as a new ID. I saw this happen at least once on the test clip with someone hidden behind the bleachers for a few seconds.
- **Crowded scenes** — the test clip's seating area has 6-8 people at once and tracking held up fine, though DeepSORT gets noticeably slower per frame as the number of people in view goes up, since it runs an embedding pass per person.
- **Camera/codec issues** — `VideoWriter` tries a few different codecs (mp4v, avc1, XVID, MJPG) until one actually works, instead of failing on the first one that isn't supported on a given machine.
- **Duplicate/repeated alerts** — a cooldown period stops the same person re-triggering the same event in the same zone every frame while they're still standing there. Intrusion fires once per visit, loitering once per stationary period.
- **Low light** — I didn't get a full test run of this in with real numbers, but based on how YOLOv8n behaves, I'd expect detection confidence and recall to drop noticeably on dark/IR-style footage, since the base model is trained mostly on well-lit COCO images. This is the one edge case I'd want to validate properly with more time, ideally against real low-light CCTV footage rather than my gym clip.

## Known limitations

- Re-identification is bounded by `max_age` — it's not true long-term re-ID. Someone gone for a minute is a new person as far as the system is concerned.
- The loitering/intrusion logic is purely geometric and temporal. It flags patterns (stayed still, entered an area), not intent — a false "suspicious" flag on someone innocently sitting down is expected behavior, not a bug.
- Detection confidence directly gates whether tracking and events happen at all, so anything that hurts YOLO's confidence (low light, small/distant people, heavy occlusion) quietly suppresses everything downstream of it.
- Zones are manually drawn polygons per video. There's no automatic zone suggestion — someone has to look at a frame and decide what matters.
- I didn't fine-tune the detector on any surveillance-specific dataset. It's pretrained COCO weights, which is fine for a prototype but would need retraining on CrowdHuman or similar for a domain like heavily crowded or overhead-camera footage.
- No GPU was available for testing, so all numbers below are CPU-only. Performance would look very different on a GPU, especially with DeepSORT.

## Performance

Measured on my own machine (CPU only, no GPU) with YOLOv8n + DeepSORT, no frame skipping:

| | |
|---|---|
| Clip | 480x360, 30fps, ~75s (2254 frames) |
| Runtime | 949 seconds (~15.8 min) |
| Average FPS | 2.4 |
| Memory | flat throughout — frames are streamed and written one at a time, never buffered |

2.4 fps is slow, and DeepSORT's per-person embedding step is the reason — it's doing a small CNN forward pass for every tracked person, every frame, on CPU. Two ways to speed this up if needed:

- `--tracker bytetrack` — no embedding step at all, much faster, at the cost of re-identification.
- `--frame-skip N` — process every (N+1)th frame. Doesn't speed up per-frame work but cuts total frames processed.

On a GPU, both the detector and DeepSORT's embedder would run substantially faster since the CNN work moves off CPU — I'd expect something closer to real-time, but I didn't have hardware to confirm that number myself.