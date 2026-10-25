"""ARS v2 (Mania et al., 2018): evaluate antithetic perturbations θ ± σδ, then step along the b best directions,
θ ← θ + α / (b σ_R) Σ (r⁺ − r⁻) δ, with σ_R the spread of the 2b returns used."""

import jax
import jax.numpy as jnp

from abps.config import Config
from abps.env.policy import Policy
from abps.env.rollout import Rollout
from abps.methods.base import Evaluate, Searcher


def ascent(directions: jax.Array, returns: jax.Array, b: int) -> tuple[jax.Array, jax.Array]:
    """(1 / (b σ_R)) Σ (r⁺ − r⁻) δ over the b directions with the best max(r⁺, r⁻); zero if σ_R = 0."""
    plus, minus = jnp.split(returns, 2)
    best = jnp.argsort(-jnp.maximum(plus, minus))[:b]
    spread = jnp.std(jnp.concatenate([plus[best], minus[best]]))
    return jnp.where(spread > 0, (plus[best] - minus[best]) @ directions[best] / (b * spread), 0), spread


class ARS(Searcher):
    def __init__(self, config: Config, policy: Policy, key: jax.Array):
        super().__init__(config, policy)

    def iteration(self, evaluate: Evaluate, key: jax.Array) -> tuple[list[Rollout], dict]:
        search = self.config.search
        directions = jax.random.normal(key, (search.queries // 2, self.theta.size))
        rollout = evaluate(self.theta + search.sigma * jnp.concatenate([directions, -directions]), search.episodes)
        step, spread = ascent(directions, rollout.returns, self.config.ars.top_directions)
        self.theta = self.theta + search.learning_rate * step
        return [rollout], {"move/spread": spread}
