import unittest

from agents import AGENT_REGISTRY, build_agent, register_agent
from agents.a2c_agent import A2CAgent
from agents.base_agent import BaseAgent
from agents.qac_agent import QACAgent
from utils.config import Config


class TestAgentFactory(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_build_a2c(self):
        agent = build_agent('a2c', 4, 2, self.config)
        self.assertIsInstance(agent, A2CAgent)

    def test_build_qac(self):
        agent = build_agent('qac', 4, 2, self.config)
        self.assertIsInstance(agent, QACAgent)

    def test_build_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            build_agent('unknown', 4, 2, self.config)

    def test_register_new_algorithm(self):
        @register_agent('dummy_for_test')
        class DummyAgent(BaseAgent):
            def __init__(self, state_dim, action_dim, config):
                self.state_dim = state_dim
                self.action_dim = action_dim

            def select_action(self, state, deterministic=False):
                return 0

            def update(self, transition):
                return 0.0, 0.0

        try:
            agent = build_agent('dummy_for_test', 4, 2, self.config)
            self.assertEqual(agent.state_dim, 4)
            self.assertEqual(agent.action_dim, 2)
        finally:
            AGENT_REGISTRY.pop('dummy_for_test', None)


if __name__ == '__main__':
    unittest.main()
