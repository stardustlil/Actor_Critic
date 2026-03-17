from agents.a2c_agent import A2CAgent
from agents.qac_agent import QACAgent

AGENT_REGISTRY = {
    "a2c": A2CAgent,
    "qac": QACAgent,
}


def build_agent(algorithm: str, state_dim: int, action_dim: int, config):
    algo = algorithm.lower()
    if algo not in AGENT_REGISTRY:
        supported = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {supported}")
    return AGENT_REGISTRY[algo](state_dim, action_dim, config)
