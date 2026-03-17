import tempfile
import textwrap
import unittest

from utils.config import Config


class ConfigParsingTests(unittest.TestCase):
    def _write_yaml(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
        tmp.write(textwrap.dedent(content))
        tmp.close()
        return tmp.name

    def test_string_values_are_coerced_to_field_types(self):
        path = self._write_yaml(
            """
            env_name: CartPole-v1
            num_episodes: "200"
            gamma: "0.95"
            actor_lr: "1e-4"
            """
        )

        config = Config.from_yaml(path)

        self.assertIsInstance(config.num_episodes, int)
        self.assertEqual(config.num_episodes, 200)
        self.assertIsInstance(config.gamma, float)
        self.assertAlmostEqual(config.gamma, 0.95)
        self.assertIsInstance(config.actor_lr, float)
        self.assertAlmostEqual(config.actor_lr, 1e-4)

    def test_unknown_override_keys_are_ignored(self):
        path = self._write_yaml("env_name: CartPole-v1\n")

        config = Config.from_yaml(path, overrides={"non_existing_key": 123, "seed": "7"})

        self.assertEqual(config.seed, 7)
        self.assertFalse(hasattr(config, "non_existing_key"))


if __name__ == '__main__':
    unittest.main()
