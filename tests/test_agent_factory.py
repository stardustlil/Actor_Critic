import unittest

from agents import build_agent
from agents.a2c_agent import A2CAgent
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


if __name__ == '__main__':
    unittest.main()
