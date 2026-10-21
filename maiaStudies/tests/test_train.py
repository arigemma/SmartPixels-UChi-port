import unittest
import csv
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from maiaStudies.checkpointing import (
    capture_fixed_connections,
    restore_fixed_connections,
)
from maiaStudies.train import (
    _append_validation_history,
    _snapshot_to_cpu,
    validation_metrics,
)


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


class _ModeSensitiveClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs[:, 0] if self.training else inputs[:, 1]
        return torch.stack((-score, score), dim=1)


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

    def test_validation_history_keeps_relaxed_and_discrete_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_history.tsv"
            discrete = {"balanced_accuracy": 0.8, "cross_entropy": 0.4}
            relaxed = {"balanced_accuracy": 0.9, "cross_entropy": 0.3}
            _append_validation_history(path, 5000, 1.25, discrete, relaxed)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], "5000")
            self.assertEqual(rows[0]["discrete_balanced_accuracy"], "0.8")
            self.assertEqual(rows[0]["relaxed_balanced_accuracy"], "0.9")

    def test_validation_metrics_explicitly_selects_and_restores_mode(self):
        inputs = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
        labels = torch.tensor([1, 0])
        timing = torch.zeros(2)
        loader = DataLoader(TensorDataset(inputs, labels, timing), batch_size=2)
        model = _ModeSensitiveClassifier()
        model.train()
        discrete = validation_metrics(model, loader, torch.device("cpu"))
        self.assertTrue(model.training)
        relaxed = validation_metrics(
            model, loader, torch.device("cpu"), discrete=False
        )
        self.assertTrue(model.training)
        self.assertEqual(discrete["balanced_accuracy"], 1.0)
        self.assertEqual(relaxed["balanced_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
