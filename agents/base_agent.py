from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """统一智能体接口，方便后续扩展更多算法。"""

    @abstractmethod
    def select_action(self, state, deterministic=False):
        raise NotImplementedError

    @abstractmethod
    def update(self, transition):
        raise NotImplementedError
