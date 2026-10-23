"""Checks for the standalone NumPy forward passes."""

import unittest

import numpy as np

import bert
import gpt
import transformer


class ModelTests(unittest.TestCase):
    def assert_distribution(self, probabilities):
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-12)

    def test_forward_passes(self):
        self.assertEqual(transformer.output_probabilities.shape, (1, 37000))
        self.assertEqual(bert.masked_probabilities.shape, (2, 30522))
        self.assertEqual(bert.next_sentence_probabilities.shape, (2,))
        self.assertEqual(gpt.output_probabilities.shape, (8, 40478))
        self.assert_distribution(transformer.output_probabilities)
        self.assert_distribution(bert.masked_probabilities)
        self.assert_distribution(bert.next_sentence_probabilities)
        self.assert_distribution(gpt.output_probabilities)

    def test_gpt_cannot_read_future_tokens(self):
        probabilities = gpt.gpt(np.array([10, 20, 30, 40]))
        changed_probabilities = gpt.gpt(np.array([10, 20, 50, 60]))
        prefix_probabilities = gpt.gpt(np.array([10, 20]))
        np.testing.assert_allclose(probabilities[:2], prefix_probabilities, atol=1e-12)
        np.testing.assert_allclose(probabilities[:2], changed_probabilities[:2], atol=1e-12)
        self.assertFalse(np.allclose(probabilities[2:], changed_probabilities[2:], rtol=1e-8, atol=1e-12))

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

    def test_bert_can_read_future_tokens(self):
        tokens = bert.inputs.copy()
        before, _ = bert.bert(tokens, bert.segment_ids, np.array([0]))
        tokens[-2] = 2000
        after, _ = bert.bert(tokens, bert.segment_ids, np.array([0]))
        self.assertFalse(np.allclose(before, after, rtol=1e-8, atol=1e-12))

    def test_bert_segment_embeddings(self):
        a, _ = bert.bert(bert.inputs, np.zeros(bert.hp["seq_len"], dtype=int), bert.masked_positions)
        b, _ = bert.bert(bert.inputs, np.ones(bert.hp["seq_len"], dtype=int), bert.masked_positions)
        self.assertFalse(np.allclose(a, b, rtol=1e-8, atol=1e-12))

    def test_heads_share_word_embeddings(self):
        x = gpt.token_embedding(gpt.inputs) + gpt.position_embedding(len(gpt.inputs))
        hidden = gpt.decoder(x)
        np.testing.assert_allclose(gpt.output_probabilities, gpt.softmax(hidden @ gpt.w["E"].T))
        x = (bert.token_embedding(bert.inputs) + bert.position_embedding(len(bert.inputs))
             + bert.segment_embedding(bert.segment_ids))
        x = bert.layer_norm(x, "embed")
        hidden = bert.encoder(x)
        x = bert.gelu(hidden[bert.masked_positions] @ bert.w["W_mlm"] + bert.w["b_mlm"])
        x = bert.layer_norm(x, "mlm")
        np.testing.assert_allclose(bert.masked_probabilities, bert.softmax(x @ bert.w["E"].T + bert.w["b_vocab"]))

    def test_layer_norm(self):
        for model, sublayer in ((bert, "self"), (gpt, "self"), (transformer, "self_enc")):
            x = np.ones((2, model.hp["d_model"]))
            x[0, 0] += 0.0001
            expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + model.hp["eps"])
            expected = model.w[f"gain_{sublayer}"][0] * expected + model.w[f"bias_{sublayer}"][0]
            np.testing.assert_allclose(model.layer_norm(x, sublayer, 0), expected)

    def test_dropout_defaults_to_identity(self):
        x = np.ones((8, 8))
        for model in (bert, gpt, transformer):
            self.assertIs(model.dropout(x), x)


if __name__ == "__main__":
    unittest.main()
