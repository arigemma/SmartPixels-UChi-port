import unittest

import numpy as np

from lgnStudies.resource_accounting import (
    estimate_dense_lgn_resources,
    estimate_group_sum_luts,
)


class ResourceAccountingTest(unittest.TestCase):
    def test_identity_network_is_only_groupsum(self):
        result = estimate_dense_lgn_resources(
            layer_lut_ids=[np.array([3, 5, 3, 5, 3, 5])],
            layer_connections=[
                np.array([[0, 0, 1, 1, 2, 2], [1, 1, 2, 2, 3, 3]])
            ],
            input_width=4,
            num_classes=3,
        )
        self.assertEqual(result.simplified_logic_luts, 0)
        self.assertEqual(result.total_luts, result.group_sum_luts)

    def test_dead_code_and_constant_propagation(self):
        # First layer: AND, unused XOR, constant 1, passthrough A.
        # Second layer outputs: AND(first AND, 1), pass(first passthrough).
        result = estimate_dense_lgn_resources(
            layer_lut_ids=[
                np.array([1, 6, 15, 3]),
                np.array([1, 3, 3]),
            ],
            layer_connections=[
                np.array([[0, 1, 2, 3], [1, 2, 3, 0]]),
                np.array([[0, 3, 2], [2, 1, 0]]),
            ],
            input_width=4,
            num_classes=3,
        )
        self.assertEqual(result.reachable_binary_luts, 1)
        self.assertEqual(result.reachable_inverters, 0)
        self.assertEqual(result.reachable_luts_by_layer, (1, 0))

    def test_group_sum_cost_is_architecture_only(self):
        self.assertEqual(
            estimate_group_sum_luts(final_layer_width=12, num_classes=3),
            12,
        )
        with self.assertRaises(ValueError):
            estimate_group_sum_luts(final_layer_width=10, num_classes=3)

    def test_inverter_is_folded_into_downstream_lut(self):
        result = estimate_dense_lgn_resources(
            layer_lut_ids=[
                np.array([12, 5]),
                np.array([1, 0, 0]),
            ],
            layer_connections=[
                np.array([[0, 0], [1, 1]]),
                np.array([[0, 0, 1], [1, 1, 0]]),
            ],
            input_width=2,
            num_classes=3,
        )
        self.assertEqual(result.reachable_binary_luts, 1)
        self.assertEqual(result.reachable_inverters, 0)


if __name__ == "__main__":
    unittest.main()
