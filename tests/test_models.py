"""Checks for the standalone NumPy forward passes."""

import unittest

import numpy as np

import transformer


class ModelTests(unittest.TestCase):
    def assert_distribution(self, probabilities):
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-12)

    def test_forward_passes(self):
        self.assertEqual(transformer.output_probabilities.shape, (1, 37000))
        self.assert_distribution(transformer.output_probabilities)


    def test_transformer_connects_encoder_and_decoder(self):
        source = np.array([10, 20, 30])
        outputs_shifted_right = np.array([transformer.hp["bos_id"], 40, 50, 60])
        probabilities = transformer.transformer(source, outputs_shifted_right=outputs_shifted_right)
        self.assertEqual(probabilities.shape, (4, 37000))
        self.assert_distribution(probabilities)

        changed_source = transformer.transformer(np.array([10, 25, 30]), outputs_shifted_right)
        self.assertFalse(np.array_equal(probabilities, changed_source))
        changed_target = transformer.transformer(source, np.array([transformer.hp["bos_id"], 40, 50, 65]))
        np.testing.assert_allclose(probabilities[:3], changed_target[:3], atol=1e-12)

        first_step = transformer.transformer(source)
        explicit_bos = transformer.transformer(source, np.array([transformer.hp["bos_id"]]))
        self.assertEqual(first_step.shape, (1, 37000))
        np.testing.assert_array_equal(first_step, explicit_bos)




    def test_layer_norm(self):
        for model, sublayer in ((transformer, "self_enc"),):
            x = np.ones((2, model.hp["d_model"]))
            x[0, 0] += 0.0001
            expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + model.hp["eps"])
            expected = model.w[f"gain_{sublayer}"][0] * expected + model.w[f"bias_{sublayer}"][0]
            np.testing.assert_allclose(model.layer_norm(x, sublayer, 0), expected)

    def test_dropout_defaults_to_identity(self):
        x = np.ones((8, 8))
        for model in (transformer,):
            self.assertIs(model.dropout(x), x)


if __name__ == "__main__":
    unittest.main()
