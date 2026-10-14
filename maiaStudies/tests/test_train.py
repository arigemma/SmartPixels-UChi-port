import unittest

import torch

from maiaStudies.checkpointing import (
    capture_fixed_connections,
    restore_fixed_connections,
)
from maiaStudies.train import _snapshot_to_cpu


class _UnregisteredConnections(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        # Emulate TorchLogix releases where this tensor was not a buffer.
        object.__setattr__(self, "indices", indices)


class _LogicLayer(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.connections = _UnregisteredConnections(indices)


class _Model(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.layer = _LogicLayer(indices)


class TrainTest(unittest.TestCase):
    def test_checkpoint_snapshot_is_detached_cpu_clone(self):
        source = torch.tensor([1.0, 2.0], requires_grad=True)
        payload = {"nested": [source]}
        snapshot = _snapshot_to_cpu(payload)
        source.data.add_(10.0)
        self.assertEqual(snapshot["nested"][0].device.type, "cpu")
        self.assertFalse(snapshot["nested"][0].requires_grad)
        torch.testing.assert_close(snapshot["nested"][0], torch.tensor([1.0, 2.0]))

    def test_fixed_connections_are_restored_when_state_dict_omits_them(self):
        trained = _Model(torch.tensor([[1, 3], [2, 0]]))
        saved = capture_fixed_connections(trained)
        rebuilt = _Model(torch.tensor([[0, 1], [3, 2]]))
        rebuilt.load_state_dict(trained.state_dict(), strict=True)
        self.assertFalse(
            torch.equal(
                rebuilt.layer.connections.indices,
                trained.layer.connections.indices,
            )
        )
        restore_fixed_connections(rebuilt, saved)
        torch.testing.assert_close(
            rebuilt.layer.connections.indices,
            trained.layer.connections.indices,
        )


if __name__ == "__main__":
    unittest.main()
