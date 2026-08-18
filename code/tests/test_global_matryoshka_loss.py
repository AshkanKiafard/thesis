import unittest

import torch

from core.constants import ActivationFunc
from core.embeddings import DistanceMetric
from finetune.astar_training_core import MatryoshkaAStarLoss


class DummyModel:
    def get_sentence_embedding_dimension(self):
        return 4


class GlobalMatryoshkaLossTests(unittest.TestCase):
    def build_loss(self, normalize=False):
        loss = MatryoshkaAStarLoss(
            model=DummyModel(),
            cls_activation_func=ActivationFunc.RELU,
            cls_distance_metric=DistanceMetric.EUCLIDEAN,
            cls_normalize=normalize,
            matryoshka_dims=[4, 2],
        )
        # Full norm is two, while the smallest prefix norm is one. This proves
        # the one global regularizer uses the complete embedding.
        raw_embedding = torch.tensor([[1.0, 0.0, 3.0 ** 0.5, 0.0]])
        loss.get_raw_embeddings = lambda _texts: raw_embedding.clone()
        return loss

    def test_regularization_is_applied_once_to_full_embeddings(self):
        loss = self.build_loss(normalize=False)

        value = loss([["c"], ["e"], ["p"], ["n"]])

        # Ranking loss is zero. Four full vectors of norm two contribute one
        # each, once globally. Per-prefix regularization would return eight.
        self.assertEqual(value.item(), 4.0)

    def test_dimensional_task_losses_are_summed_without_self_normalizing(self):
        loss = self.build_loss(normalize=False)
        loss.compute_sub_loss = (
            lambda c_emb, _e_emb, _p_emb, _n_emb: c_emb.new_tensor(c_emb.shape[1])
        )

        value = loss([["c"], ["e"], ["p"], ["n"]])

        # Dimensional terms are 4 + 2, plus the one global regularizer of 4.
        self.assertEqual(value.item(), 10.0)

    def test_explicit_normalization_disables_the_existing_regularizer(self):
        loss = self.build_loss(normalize=True)
        loss.compute_sub_loss = (
            lambda c_emb, _e_emb, _p_emb, _n_emb: c_emb.new_tensor(c_emb.shape[1])
        )

        value = loss([["c"], ["e"], ["p"], ["n"]])

        self.assertEqual(value.item(), 6.0)


if __name__ == "__main__":
    unittest.main()
