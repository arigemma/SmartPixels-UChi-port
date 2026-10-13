import unittest

import torch

from maiaStudies.train import _snapshot_to_cpu


class TrainTest(unittest.TestCase):
    def test_checkpoint_snapshot_is_detached_cpu_clone(self):
        source = torch.tensor([1.0, 2.0], requires_grad=True)
        payload = {"nested": [source]}
        snapshot = _snapshot_to_cpu(payload)
        source.data.add_(10.0)
        self.assertEqual(snapshot["nested"][0].device.type, "cpu")
        self.assertFalse(snapshot["nested"][0].requires_grad)
        torch.testing.assert_close(snapshot["nested"][0], torch.tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
