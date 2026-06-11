from dataclasses import dataclass

from .geometry import (
    box_center,
    box_iou,
    normalized_center_distance,
)


@dataclass
class Track:
    track_id: int
    class_id: int
    bbox: tuple
    center: tuple
    previous_center: tuple | None = None
    hits: int = 1
    missed: int = 0
    confidence_sum: float = 0.0
    max_confidence: float = 0.0

    @property
    def average_confidence(self):
        return self.confidence_sum / max(self.hits, 1)


class IoUTracker:
    def __init__(
        self,
        iou_threshold=0.3,
        max_age=30,
        center_distance_threshold=1.0,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.center_distance_threshold = float(center_distance_threshold)
        self.next_id = 1
        self.tracks = {}

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections):
        for track in self.tracks.values():
            track.missed += 1

        candidates = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                iou = box_iou(track.bbox, detection.xyxy)
                if iou >= self.iou_threshold:
                    candidates.append((2.0 + iou, track_id, detection_index))
                    continue

                distance = normalized_center_distance(track.bbox, detection.xyxy)
                if (
                    self.center_distance_threshold > 0
                    and distance <= self.center_distance_threshold
                ):
                    score = 1.0 - distance / self.center_distance_threshold
                    candidates.append((score, track_id, detection_index))

        matched_tracks = set()
        matched_detections = set()
        assignments = [None] * len(detections)
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            track = self.tracks[track_id]
            track.previous_center = track.center
            track.bbox = detections[detection_index].xyxy
            track.center = box_center(track.bbox)
            track.hits += 1
            track.missed = 0
            track.confidence_sum += detections[detection_index].confidence
            track.max_confidence = max(
                track.max_confidence,
                detections[detection_index].confidence,
            )
            assignments[detection_index] = track_id

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                class_id=detection.class_id,
                bbox=detection.xyxy,
                center=box_center(detection.xyxy),
                confidence_sum=detection.confidence,
                max_confidence=detection.confidence,
            )
            assignments[detection_index] = track_id

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.max_age
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]
        return assignments


def update_tracks(tracker, detections, min_confidence):
    selected_indices = [
        index
        for index, detection in enumerate(detections)
        if detection.confidence >= min_confidence
    ]
    selected_detections = [detections[index] for index in selected_indices]
    selected_assignments = tracker.update(selected_detections)
    assignments = [None] * len(detections)
    for detection_index, track_id in zip(selected_indices, selected_assignments):
        assignments[detection_index] = track_id
    return assignments
