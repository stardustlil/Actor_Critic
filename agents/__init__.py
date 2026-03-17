from typing import Callable, Dict, Type

from agents.base_agent import BaseAgent


AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def register_agent(name: str) -> Callable[[Type[BaseAgent]], Type[BaseAgent]]:
    """Register a new agent class into global registry.

    Usage:
        @register_agent("ppo")
        class PPOAgent(BaseAgent):
            ...
    """

    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        key = name.lower().strip()
        if not key:
            raise ValueError("Agent name cannot be empty.")
        if key in AGENT_REGISTRY:
            raise ValueError(f"Duplicate agent registration for '{key}'.")
        AGENT_REGISTRY[key] = cls
        return cls

    return decorator


def build_agent(algorithm: str, state_dim: int, action_dim: int, config):
    algo = algorithm.lower().strip()
    if algo not in AGENT_REGISTRY:
        supported = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {supported}")
    return AGENT_REGISTRY[algo](state_dim, action_dim, config)


from agents.a2c_agent import A2CAgent  # noqa: E402,F401
from agents.qac_agent import QACAgent  # noqa: E402,F401
