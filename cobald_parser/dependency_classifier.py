from typing import override
from copy import deepcopy

import numpy as np

import torch
from torch import nn
from torch import Tensor, FloatTensor, BoolTensor, LongTensor
import torch.nn.functional as F

from transformers.activations import ACT2FN

from cobald_parser.bilinear_matrix_attention import BilinearMatrixAttention
from cobald_parser.chu_liu_edmonds import decode_mst
from cobald_parser.utils import pairwise_mask, replace_masked_values


class DependencyHeadBase(nn.Module):
    """
    Base class for scoring arcs and relations between tokens in a dependency tree/graph.
    """

    def __init__(self, hidden_size: int, n_rels: int):
        super().__init__()

        self.arc_attention = BilinearMatrixAttention(
            hidden_size,
            hidden_size,
            use_input_biases=True,
            n_labels=1
        )
        self.rel_attention = BilinearMatrixAttention(
            hidden_size,
            hidden_size,
            use_input_biases=True,
            n_labels=n_rels
        )

    def forward(
        self,
        h_arc_head: Tensor,        # [batch_size, seq_len, hidden_size]
        h_arc_dep: Tensor,         # ...
        h_rel_head: Tensor,        # ...
        h_rel_dep: Tensor,         # ...
        gold_arcs: LongTensor,     # [batch_size, seq_len, seq_len]
        null_mask: BoolTensor,     # [batch_size, seq_len]
        padding_mask: BoolTensor   # [batch_size, seq_len]
    ) -> dict[str, Tensor]:
        
        # Score arcs.
        # s_arc[:, i, j] = score of edge i -> j.
        s_arc = self.arc_attention(h_arc_head, h_arc_dep)
        # Mask undesirable values (padding, nulls, etc.) with -inf.
        mask2d = pairwise_mask(null_mask & padding_mask)
        replace_masked_values(s_arc, mask2d, replace_with=-1e8)
        # Score arcs' relations.
        # [batch_size, seq_len, seq_len, num_labels]
        s_rel = self.rel_attention(h_rel_head, h_rel_dep).permute(0, 2, 3, 1)

        # Calculate loss.
        loss = 0.0
        if gold_arcs is not None:
            loss += self.calc_arc_loss(s_arc, gold_arcs)
            loss += self.calc_rel_loss(s_rel, gold_arcs)

        # Predict arcs based on the scores.
        # [batch_size, seq_len, seq_len]
        pred_arcs_matrix = self.predict_arcs(s_arc, null_mask, padding_mask)
        # [batch_size, seq_len, seq_len]
        pred_rels_matrix = self.predict_rels(s_rel)
        # [n_pred_arcs, 4]
        preds_combined = self.combine_arcs_rels(pred_arcs_matrix, pred_rels_matrix)
        return {
            'preds': preds_combined,
            'loss': loss
        }

    @staticmethod
    def calc_arc_loss(
        s_arc: Tensor,         # [batch_size, seq_len, seq_len]
        gold_arcs: LongTensor  # [n_arcs, 4]
    ) -> Tensor:
        """Calculate arc loss."""
        raise NotImplementedError

    @staticmethod
    def calc_rel_loss(
        s_rel: Tensor,         # [batch_size, seq_len, seq_len, num_labels]
        gold_arcs: LongTensor  # [n_arcs, 4]
    ) -> Tensor:
        batch_idxs, arcs_from, arcs_to, rels = gold_arcs.T
        return F.cross_entropy(s_rel[batch_idxs, arcs_from, arcs_to], rels)
    
    def predict_arcs(
        self,
        s_arc: Tensor,           # [batch_size, seq_len, seq_len]
        null_mask: BoolTensor,   # [batch_size, seq_len]
        padding_mask: BoolTensor # [batch_size, seq_len]
    ) -> LongTensor:
        """Predict arcs from scores."""
        raise NotImplementedError
    
    def predict_rels(
        self,
        s_rel: FloatTensor
    ) -> LongTensor:
        return s_rel.argmax(dim=-1).long()
    
    @staticmethod
    def combine_arcs_rels(
        pred_arcs: LongTensor,
        pred_rels: LongTensor
    ) -> LongTensor:
        """Select relations towards predicted arcs."""
        assert pred_arcs.shape == pred_rels.shape
        # Get indices where arcs exist
        indices = pred_arcs.nonzero(as_tuple=True)
        batch_idxs, from_idxs, to_idxs = indices
        # Get corresponding relation types
        rel_types = pred_rels[batch_idxs, from_idxs, to_idxs]
        # Stack as [batch_idx, from_idx, to_idx, rel_type]
        return torch.stack([batch_idxs, from_idxs, to_idxs, rel_types], dim=1)


class DependencyHead(DependencyHeadBase):
    """
    Basic UD syntax specialization that predicts single edge for each token.
    """

    @override
    def predict_arcs(
        self,
        s_arc: Tensor,           # [batch_size, seq_len, seq_len]
        null_mask: BoolTensor,   # [batch_size, seq_len]
        padding_mask: BoolTensor # [batch_size, seq_len, seq_len]
    ) -> Tensor:

        if self.training:
            # During training, use fast greedy decoding.
            # - [batch_size, seq_len]
            pred_arcs_seq = s_arc.argmax(dim=1)
        else:
            # FIXME
            # During inference, decode Maximum Spanning Tree.
            # pred_arcs_seq = self._mst_decode(s_arc, padding_mask)
            pred_arcs_seq = s_arc.argmax(dim=1)

        # Upscale arcs sequence of shape [batch_size, seq_len]
        # to matrix of shape [batch_size, seq_len, seq_len].
        pred_arcs = F.one_hot(pred_arcs_seq, num_classes=pred_arcs_seq.size(1)).long().transpose(1, 2)
        # Apply mask one more time (even though s_arc is already masked),
        # because argmax erases information about masked values.
        mask2d = pairwise_mask(null_mask & padding_mask)
        replace_masked_values(pred_arcs, mask2d, replace_with=0)
        return pred_arcs

    def _mst_decode(
        self,
        s_arc: Tensor,    # [batch_size, seq_len, seq_len]
        padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        
        batch_size = s_arc.size(0)
        device = s_arc.device
        s_arc = s_arc.cpu()

        # Convert scores to probabilities, as `decode_mst` expects non-negative values.
        arc_probs = nn.functional.softmax(s_arc, dim=1)

        # `decode_mst` knows nothing about UD and ROOT, so we have to manually
        # zero probabilities of arcs leading to ROOT to make sure ROOT is a source node
        # of a graph.

        # Decode ROOT positions from diagonals.
        # shape: [batch_size]
        root_idxs = arc_probs.diagonal(dim1=1, dim2=2).argmax(dim=-1)
        # Zero out arcs leading to ROOTs.
        arc_probs[torch.arange(batch_size), :, root_idxs] = 0.0

        pred_arcs = []
        for sample_idx in range(batch_size):
            energy = arc_probs[sample_idx]
            length = padding_mask[sample_idx].sum()
            heads = decode_mst(energy, length)
            # Some nodes may be isolated. Pick heads greedily in this case.
            heads[heads <= 0] = s_arc[sample_idx].argmax(dim=1)[heads <= 0]
            pred_arcs.append(heads)

        # shape: [batch_size, seq_len]
        pred_arcs = torch.from_numpy(np.stack(pred_arcs)).long().to(device)
        return pred_arcs

    @staticmethod
    @override
    def calc_arc_loss(
        s_arc: Tensor,         # [batch_size, seq_len, seq_len]
        gold_arcs: LongTensor  # [n_arcs, 4]
    ) -> tuple[Tensor, Tensor]:
        batch_idxs, from_idxs, to_idxs, _ = gold_arcs.T
        return F.cross_entropy(s_arc[batch_idxs, :, to_idxs], from_idxs)


class MultiDependencyHead(DependencyHeadBase):
    """
    Enhanced UD syntax specialization that predicts multiple edges for each token.
    """

    @override
    def predict_arcs(
        self,
        s_arc: Tensor,           # [batch_size, seq_len, seq_len]
        null_mask: BoolTensor,   # [batch_size, seq_len]
        padding_mask: BoolTensor # [batch_size, seq_len]
    ) -> Tensor:
        # Convert scores to probabilities.
        arc_probs = torch.sigmoid(s_arc)
        # Find confident arcs (with prob > 0.5).
        return arc_probs.round().long()

    @staticmethod
    @override
    def calc_arc_loss(
        s_arc: Tensor,         # [batch_size, seq_len, seq_len]
        gold_arcs: LongTensor  # [n_arcs, 4]
    ) -> Tensor:
        batch_idxs, from_idxs, to_idxs, _ = gold_arcs.T
        # Gold arcs but as a matrix, where matrix[i, arcs_from, arc_to] = 1.0 if arcs is present.
        gold_arcs_matrix = torch.zeros_like(s_arc)
        gold_arcs_matrix[batch_idxs, from_idxs, to_idxs] = 1.0
        # Padded arcs's logits are huge negative values that doesn't contribute to the loss.
        return F.binary_cross_entropy_with_logits(s_arc, gold_arcs_matrix)


class DependencyClassifier(nn.Module):
    """
    Dozat and Manning's biaffine dependency classifier.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_rels_ud: int,
        n_rels_eud: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()

        self.arc_dep_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            ACT2FN[activation],
            nn.Dropout(dropout)
        )
        # All mlps are equal.
        self.arc_head_mlp = deepcopy(self.arc_dep_mlp)
        self.rel_dep_mlp = deepcopy(self.arc_dep_mlp)
        self.rel_head_mlp = deepcopy(self.arc_dep_mlp)

        self.dependency_head_ud = DependencyHead(hidden_size, n_rels_ud) if n_rels_ud > 0 else None
        self.dependency_head_eud = MultiDependencyHead(hidden_size, n_rels_eud) if n_rels_eud > 0 else None

    def forward(
        self,
        embeddings: Tensor,    # [batch_size, seq_len, embedding_size]
        gold_ud: Tensor,       # [n_ud_arcs, 4]
        gold_eud: Tensor,      # [n_eud_arcs, 4]
        null_mask: Tensor,     # [batch_size, seq_len]
        padding_mask: Tensor   # [batch_size, seq_len]
    ) -> dict[str, Tensor]:

        # - [batch_size, seq_len, hidden_size]
        h_arc_head = self.arc_head_mlp(embeddings)
        h_arc_dep = self.arc_dep_mlp(embeddings)
        h_rel_head = self.rel_head_mlp(embeddings)
        h_rel_dep = self.rel_dep_mlp(embeddings)

        result = {}

        # Share the h vectors between dependency and multi-dependency heads.
        if self.dependency_head_ud is not None:
            output_ud = self.dependency_head_ud(
                h_arc_head,
                h_arc_dep,
                h_rel_head,
                h_rel_dep,
                gold_arcs=gold_ud,
                null_mask=null_mask,
                padding_mask=padding_mask
            )
            result['preds_ud'] = output_ud["preds"]
            result['loss_ud'] = output_ud["loss"]

        if self.dependency_head_eud is not None:
            output_eud = self.dependency_head_eud(
                h_arc_head,
                h_arc_dep,
                h_rel_head,
                h_rel_dep,
                gold_arcs=gold_eud,
                # Ignore null mask in E-UD
                null_mask=torch.ones_like(padding_mask),
                padding_mask=padding_mask
            )
            result['preds_eud'] = output_eud["preds"]
            result['loss_eud'] = output_eud["loss"]

        return result
