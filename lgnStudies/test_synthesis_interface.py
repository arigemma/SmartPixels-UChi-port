import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from lgnStudies.synthesis_interface import (
    export_dense_lgn_stage,
    make_portable_dense_lgn,
)


class FakeConnections:
    def __init__(self):
        self.indices = torch.tensor([0, 1])
        self.device = "cpu"


class FakeLogicLayer(torch.nn.Module):
    out_dim = 2

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.connections = FakeConnections()
        self.device = "cpu"

    def forward(self, values):
        return values[:, self.connections.indices] * self.scale


class FakeGroupSum(torch.nn.Module):
    tau = 40.0

    def forward(self, values):
        return values


class FakeDenseModel(torch.nn.Module):
    n_bits = 100

    def __init__(self):
        super().__init__()
        self.bin = torch.nn.Identity()
        self.layers = torch.nn.ModuleList([FakeLogicLayer()])
        self.group_sum = FakeGroupSum()

    def forward(self, values):
        values = self.bin(values)
        for layer in self.layers:
            values = layer(values)
        return self.group_sum(values)


class SynthesisInterfaceTest(unittest.TestCase):
    def test_portable_model_matches_original_and_preserves_connections(self):
        model = FakeDenseModel().eval()
        portable = make_portable_dense_lgn(model)
        values = torch.tensor([[0.25, 0.75]])
        self.assertTrue(torch.equal(model(values), portable(values)))
        self.assertTrue(
            torch.equal(portable[1].connections.indices, torch.tensor([0, 1]))
        )

    def test_export_layout_matches_synthesis_catalog(self):
        with TemporaryDirectory() as temporary:
            target = export_dense_lgn_stage(
                FakeDenseModel(),
                Path(temporary),
                "s09_p1460M_tau40_step000200",
                model_alias="s09_p1460M_tau40",
                registry_name="registry-entry",
                training_step=200,
                balanced_accuracy=0.8,
            )
            self.assertTrue((target / "best_model.importable.pth").is_file())
            self.assertTrue((target / "SHA256SUMS").is_file())
            metadata = json.loads((target / "model.json").read_text())
            self.assertEqual(metadata["model_set"], "model2")
            self.assertEqual(
                metadata["training_metadata"]["training_step"], 200
            )
            loaded = torch.load(
                target / "best_model.importable.pth",
                map_location="cpu",
                weights_only=False,
            )
            self.assertIsInstance(loaded, torch.nn.Sequential)


if __name__ == "__main__":
    unittest.main()
