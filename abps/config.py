"""An experiment, whole, from its config file `configs/<method>/<task>.yaml`. The settings follow the paper's runs
with a reduced critic-training budget; `corrected` replaces the three deviations of the released ABS code
(DEVIATIONS.md B1, B2, B4) by what the paper describes."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import yaml

CONFIGS = Path(__file__).parents[1] / "configs"


@dataclass(frozen=True, kw_only=True)
class Env:
    """A Gymnasium task, the linear policies searched on it, and the objective the search maximizes."""

    name: str
    kwargs: dict[str, bool | float | str] = field(default_factory=dict)  # passed to `gymnasium.make`
    horizon: int = 1000
    discount: float  # of the training objective; the learning curve reports undiscounted returns
    reward_scale: float  # training rewards are divided by it
    normalize_obs: bool
    features: Literal["identity", "pendulum"] = "identity"
    action_scale: float = 1.0  # a = action_scale · tanh(φ(s) W)


@dataclass(frozen=True, kw_only=True)
class Search:
    """What every method shares: an iteration evaluates `queries` policies around the center for `episodes` each,
    then moves the center."""

    iterations: int
    queries: int
    episodes: int = 1
    sigma: float  # scale of random queries around the center
    learning_rate: float


@dataclass(frozen=True, kw_only=True)
class GP:
    """Local Bayesian optimization (ABS, MPD): the GP and its center."""

    n_max: int  # the GP conditions on the n_max most recent evaluations
    center_episodes: int  # the center is evaluated anew every iteration
    lengthscale_bounds: tuple[float, float]
    constant_mean: bool  # MPD: a constant prior mean in [-1, 1], fitted with the other hyperparameters
    fit_mean: Literal["average", "pointwise"] = "pointwise"  # B1: the released ABS fitted to the average prior mean
    restarts: int = 32
    maxiter: int = 500
    tolerance: float = 1e-3
    query_bound: float = 0.1  # queries are sought in θ + [-bound, bound]^d


@dataclass(frozen=True, kw_only=True)
class Critic:
    """ABS's ensemble of TD3-style twin critics of the center policy (paper Appendix B)."""

    aggregation: Literal["mean", "softmax", "max"]  # how validation R² weights the critics; B4: the runs used mean
    reset_worst: bool  # B2: re-initializing the worst critic at a new center never ran in the paper
    n_critics: int = 5
    hidden: tuple[int, ...] = (256, 256)
    dropout: float = 0.01
    learning_rate: float = 3e-4
    batch_size: int = 256
    polyak: float = 0.005
    target_every: int = 2
    steps: int = 500  # TD steps after each query; twice as many after a new center
    first_steps: int = 1000
    capacity: int = 1_000_000


@dataclass(frozen=True, kw_only=True)
class ARS:
    top_directions: int


@dataclass(frozen=True, kw_only=True)
class Config:
    method: Literal["abs", "mpd", "ars"]
    task: str
    env: Env
    search: Search
    gp: GP | None = None  # ABS and MPD
    critic: Critic | None = None  # ABS
    ars: ARS | None = None  # ARS

    @classmethod
    def read(cls, method: str, task: str) -> "Config":
        values = yaml.safe_load((CONFIGS / method / f"{task}.yaml").read_text())
        return cls.load({"method": method, "task": task, **values})

    @classmethod
    def load(cls, values: dict) -> "Config":
        """A config from its plain form, as a config file or a checkpoint's metadata holds it."""
        sections = {"env": Env, "search": Search, "gp": GP, "critic": Critic, "ars": ARS}
        built = {name: kind(**hashable(values[name])) for name, kind in sections.items() if values.get(name)}
        return cls(**values | built)


def hashable(section: dict) -> dict:
    """YAML and JSON write tuples as lists; `jax.jit` needs configs hashable."""
    return {key: tuple(value) if isinstance(value, list) else value for key, value in section.items()}


def corrected(config: Config) -> Config:
    """ABS as the paper describes it: the GP fits the pointwise prior mean, the worst critic is reset at each new
    center, and the critics are weighted by softmax(R²)."""
    if config.method != "abs":
        return config
    gp = replace(config.gp, fit_mean="pointwise")
    return replace(config, gp=gp, critic=replace(config.critic, reset_worst=True, aggregation="softmax"))


def presets() -> dict[str, Config]:
    """Every experiment config, by `<method>/<task>`."""
    paths = sorted(CONFIGS.glob("*/*.yaml"))
    return {f"{path.parent.name}/{path.stem}": Config.read(path.parent.name, path.stem) for path in paths}
