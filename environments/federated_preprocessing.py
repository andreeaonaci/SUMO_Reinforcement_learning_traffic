

import numpy as np
import time


class LaneEncoder:
    """Canonical lane feature schema shared by every city."""

    FEATURES = [
        ("queue", lambda l, n: l.queue / n.max_queue),
        ("waiting_time", lambda l, n: l.waiting_time / n.max_wait),
        ("occupancy", lambda l, n: l.occupancy),
        ("speed", lambda l, n: l.speed / n.max_speed),
        ("is_left", lambda l, n: float(l.is_left)),
        ("is_straight", lambda l, n: float(l.is_straight)),
        ("is_right", lambda l, n: float(l.is_right)),
    ]

    GLOBAL_FEATURES = [
        ("current_phase", lambda phase, elapsed, yellow: phase / 16.0),
        ("elapsed_green", lambda phase, elapsed, yellow: elapsed / 120.0),
        ("yellow_time", lambda phase, elapsed, yellow: yellow / 10.0),
    ]


class LaneExtractor:
    """Override extract() for each city/environment."""

    def __init__(self, env):
        self.env = env

    def extract(self):
        raise NotImplementedError


class LaneNormalizer:
    def __init__(self, max_queue=50, max_wait=300,
                 max_speed=20.0, encoder=LaneEncoder):
        self.max_queue = max_queue
        self.max_wait = max_wait
        self.max_speed = max_speed
        self.encoder = encoder

    def normalize(self, lane):
        return np.asarray(
            [fn(lane, self) for _, fn in self.encoder.FEATURES],
            dtype=np.float32
        )


class LaneSorter:
    def __init__(self, key=None):
        self.key = key or (
            lambda l: (l.queue, l.waiting_time, l.occupancy)
        )

    def sort(self, lanes):
        return sorted(lanes, key=self.key, reverse=True)


class TopKEncoder:
    def __init__(self, normalizer, max_lanes=16):
        self.normalizer = normalizer
        self.max_lanes = max_lanes
        self.features_per_lane = len(
            self.normalizer.encoder.FEATURES
        )
        self.output_dim = (
            self.max_lanes * self.features_per_lane +
            len(self.normalizer.encoder.GLOBAL_FEATURES)
        )

    def encode(self, lanes, current_phase,
               elapsed_green, yellow_time=0):

        lanes = lanes[:self.max_lanes]
        features = []

        for lane in lanes:
            features.extend(self.normalizer.normalize(lane))

        while len(lanes) < self.max_lanes:
            features.extend(
                np.zeros(
                    self.features_per_lane,
                    dtype=np.float32
                )
            )
            lanes.append(None)

        for _, fn in self.normalizer.encoder.GLOBAL_FEATURES:
            features.append(
                fn(current_phase, elapsed_green, yellow_time)
            )

        return np.asarray(features, dtype=np.float32)


class ActionMapper:
    def __init__(self, mapping):
        self.mapping = mapping

    def map(self, action):
        return self.mapping.get(int(action), 0)


class FederatedWrapper:
    def __init__(self, env, extractor,
                 sorter, encoder, mapper):
        self.env = env
        self.extractor = extractor
        self.sorter = sorter
        self.encoder = encoder
        self.mapper = mapper

    def _state(self):
        lanes, phase, elapsed = self.extractor.extract()
        lanes = self.sorter.sort(lanes)
        return self.encoder.encode(lanes, phase, elapsed)

    def reset(self):
        if hasattr(self.env, "episode") and self.env.episode > 0:
            time.sleep(2)
        ret = self.env.reset()
        if isinstance(ret, tuple):
            ret = ret[0]
        return self._state()

    def step(self, action):
        local_action = self.mapper.map(action)
        _, reward, done, info = self.env.step(local_action)
        return self._state(), reward, done, info

    def close(self):
        self.env.close()
