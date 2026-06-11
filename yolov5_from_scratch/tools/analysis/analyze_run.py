import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def analyze_run(run_dir):
    run_dir = Path(run_dir)
    detections_path = run_dir / "detections.csv"
    counts_path = run_dir / "counts.csv"
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detections CSV: {detections_path}")

    rows = []
    class_stats = defaultdict(
        lambda: {
            "detections": 0,
            "confidence_sum": 0.0,
            "min_confidence": 1.0,
            "max_confidence": 0.0,
            "tracks": set(),
        }
    )
    track_lengths = Counter()
    crossings = Counter()

    with open(detections_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            class_name = row["class_name"]
            confidence = float(row["confidence"])
            track_id = row["track_id"]
            stats = class_stats[class_name]
            stats["detections"] += 1
            stats["confidence_sum"] += confidence
            stats["min_confidence"] = min(stats["min_confidence"], confidence)
            stats["max_confidence"] = max(stats["max_confidence"], confidence)
            if track_id:
                stats["tracks"].add(track_id)
                track_lengths[track_id] += 1
            if row["crossing_direction"]:
                crossings[(class_name, row["crossing_direction"])] += 1

    lengths = list(track_lengths.values())
    class_summary = {}
    for class_name, stats in sorted(class_stats.items()):
        detections = stats["detections"]
        class_summary[class_name] = {
            "detections": detections,
            "tracks": len(stats["tracks"]),
            "average_confidence": stats["confidence_sum"] / max(detections, 1),
            "minimum_confidence": stats["min_confidence"],
            "maximum_confidence": stats["max_confidence"],
        }

    saved_counts = []
    if counts_path.exists():
        with open(counts_path, "r", encoding="utf-8", newline="") as f:
            saved_counts = list(csv.DictReader(f))

    return {
        "run_dir": str(run_dir.resolve()),
        "detections": len(rows),
        "tracked_detections": sum(track_lengths.values()),
        "untracked_detections": len(rows) - sum(track_lengths.values()),
        "unique_tracks": len(track_lengths),
        "track_length": {
            "median": statistics.median(lengths) if lengths else 0,
            "p90": percentile(lengths, 0.90),
            "maximum": max(lengths, default=0),
            "one_frame_tracks": sum(length == 1 for length in lengths),
            "tracks_at_most_10_frames": sum(length <= 10 for length in lengths),
        },
        "crossings": {
            f"{class_name}:{direction}": count
            for (class_name, direction), count in sorted(crossings.items())
        },
        "saved_counts": saved_counts,
        "classes": class_summary,
    }


def print_summary(summary):
    print(f"run_dir: {summary['run_dir']}")
    print(f"detections: {summary['detections']}")
    print(f"tracked_detections: {summary['tracked_detections']}")
    print(f"untracked_detections: {summary['untracked_detections']}")
    print(f"unique_tracks: {summary['unique_tracks']}")

    track_length = summary["track_length"]
    print(
        "track_length: "
        f"median={track_length['median']} "
        f"p90={track_length['p90']} "
        f"max={track_length['maximum']} "
        f"one_frame={track_length['one_frame_tracks']} "
        f"<=10_frames={track_length['tracks_at_most_10_frames']}"
    )

    print("classes:")
    for class_name, stats in summary["classes"].items():
        print(
            f"  {class_name}: detections={stats['detections']} "
            f"tracks={stats['tracks']} "
            f"avg_conf={stats['average_confidence']:.3f} "
            f"range=[{stats['minimum_confidence']:.3f}, "
            f"{stats['maximum_confidence']:.3f}]"
        )

    print("crossings:")
    if summary["crossings"]:
        for key, value in summary["crossings"].items():
            print(f"  {key} = {value}")
    else:
        print("  none")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a predict.py output directory"
    )
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = analyze_run(args.run_dir)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
