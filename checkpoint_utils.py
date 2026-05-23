import os


LATEST_CHECKPOINT = "snooker_ppo_latest.pt"
BEST_CHECKPOINT = "snooker_ppo_best.pt"


def package_dir():
    return os.path.dirname(os.path.abspath(__file__))


def project_root():
    return os.path.abspath(os.path.join(package_dir(), os.pardir))


def canonical_checkpoint_dir():
    return os.path.join(project_root(), "checkpoints")


def canonical_checkpoint_path(filename=LATEST_CHECKPOINT):
    return os.path.join(canonical_checkpoint_dir(), filename)


def checkpoint_search_paths(explicit_path=None):
    if explicit_path:
        return [os.path.abspath(explicit_path)]

    paths = []
    for base in (
        canonical_checkpoint_dir(),
        os.path.join(package_dir(), "checkpoints"),
        os.path.abspath("checkpoints"),
    ):
        for filename in (BEST_CHECKPOINT, LATEST_CHECKPOINT):
            path = os.path.abspath(os.path.join(base, filename))
            if path not in paths:
                paths.append(path)
    return paths


def _as_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math_is_finite(value) else default


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def math_is_finite(value):
    return value == value and value not in (float("inf"), float("-inf"))


def checkpoint_quality_score(payload):
    if not isinstance(payload, dict):
        return -1.0

    games = max(0, _as_int(payload.get("games"), 0))
    wins = max(0, _as_int(payload.get("wins"), 0))
    pots = max(0, _as_int(payload.get("pots"), 0))
    hits = max(0, _as_int(payload.get("hits"), 0))
    ball_count = max(0, min(15, _as_int(payload.get("ball_count"), 0)))
    stage = max(0, _as_int(payload.get("curriculum_stage"), 0))
    best_eval = max(0.0, min(1.0, _as_float(payload.get("best_eval_win_rate"), 0.0)))

    win_rate = wins / games if games else 0.0
    pots_per_game = pots / games if games else 0.0
    hits_per_game = hits / games if games else 0.0

    return (
        best_eval * 1000.0
        + win_rate * 120.0
        + min(pots_per_game, 4.0) * 25.0
        + min(hits_per_game, 30.0) * 0.5
        + ball_count * 2.0
        + stage * 10.0
    )


def checkpoint_quality_too_low(payload):
    if not isinstance(payload, dict):
        return True

    games = max(0, _as_int(payload.get("games"), 0))
    wins = max(0, _as_int(payload.get("wins"), 0))
    pots = max(0, _as_int(payload.get("pots"), 0))
    best_eval = max(0.0, min(1.0, _as_float(payload.get("best_eval_win_rate"), 0.0)))

    if best_eval >= 0.10:
        return False
    if games < 20:
        return False

    win_rate = wins / games if games else 0.0
    pots_per_game = pots / games if games else 0.0
    return win_rate < 0.05 and pots_per_game < 0.10
