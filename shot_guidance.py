import math
import random


BALL_RADIUS = 12
DEFAULT_HOLES = [
    (80, 80),
    (600, 70),
    (1120, 80),
    (80, 720),
    (600, 730),
    (1120, 720),
]


def _angle_delta(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _ball_type(ball):
    return "striped" if ball.get("is_striped", False) else "solid"


def _should_hit_8_ball(balls, player_type):
    if player_type not in ("solid", "striped"):
        return False
    for ball in balls:
        if ball.get("potted", False) or ball.get("number", 0) == 8:
            continue
        if _ball_type(ball) == player_type:
            return False
    return True


def _target_balls(obs):
    balls = [ball for ball in obs.get("balls", []) if ball.get("number", 0) != 0]
    current_player = int(obs.get("current_player", 0))
    player_types = obs.get("player_types", [None, None])
    player_type = player_types[current_player] if current_player < len(player_types) else None

    live = [ball for ball in balls if not ball.get("potted", False)]
    if _should_hit_8_ball(live, player_type):
        return [ball for ball in live if ball.get("number", 0) == 8]
    if player_type in ("solid", "striped"):
        return [
            ball for ball in live
            if ball.get("number", 0) != 8 and _ball_type(ball) == player_type
        ]
    return [ball for ball in live if ball.get("number", 0) != 8]


def _point_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-6:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _path_clear(start, end, balls, ignore_numbers, ball_radius):
    ax, ay = start
    bx, by = end
    for ball in balls:
        if ball.get("potted", False) or ball.get("number", 0) in ignore_numbers:
            continue
        distance = _point_segment_distance(ball["x"], ball["y"], ax, ay, bx, by)
        obstacle_radius = float(ball.get("radius", ball_radius))
        if distance < ball_radius + obstacle_radius + 2.0:
            return False
    return True


def _cue_ball(obs):
    for ball in obs.get("balls", []):
        if ball.get("number", 0) == 0:
            return ball
    balls = obs.get("balls", [])
    return balls[0] if balls else {"x": 300, "y": 400}


def guided_shot_from_observation(obs, width=1200, height=800, holes=None, jitter=0.0, ball_radius=None):
    """Pick a reasonable pool shot from observation geometry.

    The guide prefers clear direct pots. If no pot is visible, it falls back to
    making legal first contact with the nearest target ball.
    """
    holes = holes or DEFAULT_HOLES
    cue = _cue_ball(obs)
    ball_radius = float(ball_radius if ball_radius is not None else cue.get("radius", BALL_RADIUS))
    balls = [ball for ball in obs.get("balls", []) if ball.get("number", 0) != 0]
    targets = _target_balls(obs)
    if not targets:
        return {
            "angle": random.uniform(0, 2 * math.pi),
            "power": random.uniform(5.0, 10.0),
        }

    cue_pos = (cue["x"], cue["y"])
    candidates = []
    for target in targets:
        target_pos = (target["x"], target["y"])
        for hole in holes:
            to_hole_x = hole[0] - target["x"]
            to_hole_y = hole[1] - target["y"]
            target_to_hole = math.hypot(to_hole_x, to_hole_y)
            if target_to_hole <= 1e-6:
                continue

            ux = to_hole_x / target_to_hole
            uy = to_hole_y / target_to_hole
            ghost = (
                target["x"] - ux * ball_radius * 2.0,
                target["y"] - uy * ball_radius * 2.0,
            )
            cue_to_ghost = math.hypot(ghost[0] - cue["x"], ghost[1] - cue["y"])
            if cue_to_ghost <= ball_radius:
                continue

            if not _path_clear(cue_pos, ghost, balls, {target.get("number", 0)}, ball_radius):
                continue
            if not _path_clear(target_pos, hole, balls, {target.get("number", 0)}, ball_radius):
                continue

            shot_angle = math.atan2(ghost[1] - cue["y"], ghost[0] - cue["x"])
            contact_angle = math.atan2(target["y"] - cue["y"], target["x"] - cue["x"])
            cut_penalty = _angle_delta(shot_angle, contact_angle)
            distance_score = (cue_to_ghost + target_to_hole) / max(width, height)
            score = cut_penalty * 1.7 + distance_score
            power = max(4.5, min(18.0, 4.5 + (cue_to_ghost + target_to_hole) / 95.0))
            candidates.append((score, shot_angle, power))

    if candidates:
        _, angle, power = min(candidates, key=lambda item: item[0])
    else:
        target = min(targets, key=lambda ball: math.hypot(ball["x"] - cue["x"], ball["y"] - cue["y"]))
        distance = math.hypot(target["x"] - cue["x"], target["y"] - cue["y"])
        angle = math.atan2(target["y"] - cue["y"], target["x"] - cue["x"])
        power = max(5.0, min(13.0, 5.0 + distance / 90.0))

    if jitter:
        angle += random.uniform(-jitter, jitter)
        power += random.uniform(-1.0, 1.0) * min(1.0, jitter / 0.08)

    return {"angle": angle % (2 * math.pi), "power": max(0.0, min(25.0, power))}


def action_has_target_contact(obs, action, max_contact_distance=None):
    cue = _cue_ball(obs)
    ball_radius = float(cue.get("radius", BALL_RADIUS))
    if max_contact_distance is None:
        max_contact_distance = ball_radius * 2.05
    angle = float(action.get("angle", 0.0))
    ux = math.cos(angle)
    uy = math.sin(angle)
    targets = _target_balls(obs)
    for target in targets:
        dx = target["x"] - cue["x"]
        dy = target["y"] - cue["y"]
        projection = dx * ux + dy * uy
        if projection <= 0:
            continue
        perpendicular = abs(dx * uy - dy * ux)
        if perpendicular <= max_contact_distance:
            return True
    return False
