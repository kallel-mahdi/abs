"""What the acquisition movie shows and when. Edit here to change the timing, the captions or the legend.

The panels t = 0, 1, 2 unroll one by one. In each, the map lights up, the gradients and their cone grow, and the next
rollout appears; it then travels into the next panel as an observed point. In the last panel the rollout is run
instead: the posterior updates, the map fades, and θ steps along the most probable ascent direction.
"""

from dataclasses import dataclass

FPS, PANELS = 10, 3
# Frames per beat within a panel.
HEAT, ARROWS, QUERY, PAUSE, TRAVEL = 10, 8, 6, 10, 10
SETTLED = HEAT + ARROWS + QUERY + PAUSE  # the panel is complete
PANEL = SETTLED + TRAVEL
# Frames of the ending, in the last panel.
OBSERVE, STEP, HOLD = 12, 16, 34
FRAMES = (PANELS - 1) * PANEL + SETTLED + OBSERVE + STEP + HOLD

# One caption per beat (map, gradients, query) for each panel.
CAPTIONS = [
    ("The map scores each candidate rollout $z$ by how much it reveals about the ascent direction.",
     "Through the performance difference lemma, the critic gives the prior: the policy gradient $\\nabla_\\theta m$.",
     "Acquire its maximum: the most informative rollout."),
    ("With that return observed, the map changes.",
     "The posterior mean $\\mu_\\theta$ turns away from the critic's policy gradient.",
     "Acquire the new maximum."),
    ("Another return, another map.",
     "The posterior keeps weighing the observed returns against the critic.",
     "Acquire the next rollout."),
]
TRAVEL_CAPTION = "Run it and observe its return."
OBSERVE_CAPTION = "Run it: its return updates the posterior."
STEP_CAPTION = "Step along the most probable ascent direction."

# Legend entries, points first and then gradients, with the frame each first appears at.
LEGEND = [
    ("theta", "policy $\\theta$", 0),
    ("query", "next rollout", HEAT + ARROWS),
    ("observed", "observed rollout", PANEL),
    ("critic", "policy gradient $\\nabla_\\theta m$", HEAT),
    ("posterior", "posterior mean $\\mu_\\theta$", PANEL + HEAT),
    ("cone", "its uncertainty", HEAT),
]


@dataclass(frozen=True)
class PanelState:
    heat: float  # brightness of the map
    grow: float  # length of the gradients and radius of the cone
    query: bool  # the next rollout is shown
    ring: float  # size of the ring closing in on it
    observed: bool  # it has been run (last panel)
    update: float  # progress from this posterior to the next (last panel)
    step: float  # progress of θ along the ascent direction (last panel)


def ramp(frame: int, start: int, length: int) -> float:
    return min(max((frame - start + 1) / length, 0.0), 1.0)


def smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def active(frame: int) -> tuple[int, int]:
    """The newest panel and the frame counted from its appearance."""
    k = min(frame // PANEL, PANELS - 1)
    return k, frame - k * PANEL


def panel_state(k: int, frame: int) -> PanelState | None:
    local = frame - k * PANEL
    if local < 0:
        return None
    last = k == PANELS - 1
    update = smooth(ramp(local, SETTLED, OBSERVE)) if last else 0.0
    return PanelState(
        heat=ramp(local, 0, HEAT) * (1 - update),
        grow=ramp(local, HEAT, ARROWS),
        query=local >= HEAT + ARROWS and update <= 0.5,
        ring=1 - ramp(local, HEAT + ARROWS, QUERY),
        observed=update > 0.5,
        update=update,
        step=smooth(ramp(local, SETTLED + OBSERVE, STEP)) if last else 0.0,
    )


def travel(frame: int) -> tuple[int, float] | None:
    """The panel whose rollout is travelling to the next one, and how far it got."""
    k, local = active(frame)
    return (k, smooth(ramp(local, SETTLED, TRAVEL))) if k + 1 < PANELS and local >= SETTLED else None


def caption(frame: int) -> str:
    k, local = active(frame)
    if local < SETTLED:
        return CAPTIONS[k][0 if local < HEAT else 1 if local < HEAT + ARROWS else 2]
    if k + 1 < PANELS:
        return TRAVEL_CAPTION
    return OBSERVE_CAPTION if local < SETTLED + OBSERVE else STEP_CAPTION
