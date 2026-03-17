import tempfile
import unittest
from pathlib import Path

from utils.visualization import plot_training_curves


class TestVisualization(unittest.TestCase):
    def test_plot_training_curves_create_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'curve.png'
            result = plot_training_curves([1, 2, 3], [0.5, 0.3], [1.0, 0.7], output)
            self.assertTrue(Path(result).exists())


if __name__ == '__main__':
    unittest.main()
