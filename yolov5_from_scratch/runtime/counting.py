from collections import Counter


class LineCounter:
    def __init__(
        self,
        orientation="horizontal",
        position=0.5,
        margin=2.0,
        min_track_hits=2,
        min_confidence=0.15,
        allowed_directions=None,
    ):
        if not 0.0 <= position <= 1.0:
            raise ValueError("--line-position must be between 0 and 1")
        self.orientation = orientation
        self.position = float(position)
        self.margin = float(margin)
        self.min_track_hits = int(min_track_hits)
        self.min_confidence = float(min_confidence)
        self.allowed_directions = (
            None if allowed_directions is None else set(allowed_directions)
        )
        self.counted_tracks = set()
        self.track_sides = {}
        self.counts = Counter()

    def reset(self):
        self.counted_tracks.clear()
        self.track_sides.clear()
        self.counts.clear()

    def update(self, track, frame_shape, class_name):
        if self.orientation == "none":
            return None

        frame_h, frame_w = frame_shape[:2]
        if self.orientation == "horizontal":
            line = frame_h * self.position
            previous = (
                track.previous_center[1]
                if track.previous_center is not None
                else None
            )
            current = track.center[1]
            positive_direction = "top_to_bottom"
            negative_direction = "bottom_to_top"
        else:
            line = frame_w * self.position
            previous = (
                track.previous_center[0]
                if track.previous_center is not None
                else None
            )
            current = track.center[0]
            positive_direction = "left_to_right"
            negative_direction = "right_to_left"

        def side(value):
            if value < line - self.margin:
                return -1
            if value > line + self.margin:
                return 1
            return 0

        current_side = side(current)
        previous_side = self.track_sides.get(track.track_id)
        if previous_side is None and track.previous_center is not None:
            previous_side = side(previous)
        if current_side != 0:
            self.track_sides[track.track_id] = current_side

        if (
            track.previous_center is None
            or track.hits < self.min_track_hits
            or track.average_confidence < self.min_confidence
            or track.track_id in self.counted_tracks
            or previous_side not in (-1, 1)
            or current_side not in (-1, 1)
            or previous_side == current_side
        ):
            return None

        direction = (
            positive_direction if previous_side < current_side else negative_direction
        )
        if (
            self.allowed_directions is not None
            and direction not in self.allowed_directions
        ):
            return None

        self.counted_tracks.add(track.track_id)
        self.counts[(class_name, direction)] += 1
        return direction
