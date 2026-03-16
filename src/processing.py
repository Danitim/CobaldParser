import json
import itertools

import torch
from torch import LongTensor
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    ClassLabel
)

from src.lemmatize_helper import construct_lemma_rule
from cobald_parser.utils import pad_sequences


ROOT_HEAD = '0'

# Sentence metadata
SENT_ID = "sent_id"
TEXT = "text"

# Fields
ID = "id"
WORD = "word"
LEMMA = "lemma"
UPOS = "upos"
XPOS = "xpos"
FEATS = "feats"
HEAD = "head"
DEPREL = "deprel"
DEPS = "deps"
MISC = "misc"
DEEPSLOT = "deepslot"
SEMCLASS = "semclass"

# Transformed fields
LEMMA_RULE = "lemma_rule"
JOINT_FEATS = "joint_feats"
UD_ARC_FROM = "ud_arc_from"
UD_ARC_TO = "ud_arc_to"
UD_DEPREL = "ud_deprel"
EUD_ARC_FROM = "eud_arc_from"
EUD_ARC_TO = "eud_arc_to"
EUD_DEPREL = "eud_deprel"


def remove_range_tokens(sentence: dict) -> dict:
    """
    Remove range tokens from a sentence.
    """
    def is_range_id(idtag: str) -> bool:
        return '-' in idtag
    
    sentence_length = len(sentence[ID])
    return {
        key: [values[i]
              for i in range(sentence_length)
              if not is_range_id(sentence[ID][i])]
        for key, values in sentence.items()
        if values is not None and isinstance(values, list)
    }


def transform_fields(sentence: dict) -> dict:
    """
    Transform sentence fields:
     * turn words and lemmas into lemma rules,
     * merge upos, xpos and feats into "pos-feats",
     * encode ud syntax into a single 2d matrix,
     * same for e-ud syntax.
    """
    result = {}

    if LEMMA in sentence:
        result[LEMMA_RULE] = [
            construct_lemma_rule(word, lemma)
            if lemma is not None else None
            for word, lemma in zip(
                sentence[WORD],
                sentence[LEMMA],
                strict=True
            )
        ]
    
    if UPOS in sentence or XPOS in sentence or FEATS in sentence:
        n_tokens = len(sentence[WORD])
        upos_list = sentence.get(UPOS, [None] * n_tokens)
        xpos_list = sentence.get(XPOS, [None] * n_tokens)
        feats_list = sentence.get(FEATS, [None] * n_tokens)
        result[JOINT_FEATS] = [
            f"{upos or '_'}#{xpos or '_'}#{feats or '_'}"
            if (upos is not None or xpos is not None or feats is not None) else None
            for upos, xpos, feats in zip(
                upos_list,
                xpos_list,
                feats_list,
                strict=True
            )
        ]

    # Renumerate ids, so that tokens are enumerated from 0 and #NULLs get integer id.
    # E.g. [1, 1.1, 2] -> [0, 1, 2].
    id2idx = {token_id: token_idx for token_idx, token_id in enumerate(sentence[ID])}

    # Basic syntax.
    if HEAD in sentence and DEPREL in sentence:
        ud_arcs_from, ud_arcs_to, ud_deprels = zip(
            *[
                (
                    # Replace ROOT with self-loop, it simplifies dependency classifier
                    # implementation a lot.
                    id2idx[str(head_id)] if str(head_id) != ROOT_HEAD else id2idx[token_id],
                    id2idx[token_id],
                    deprel
                )
                # head_id indicates ID of a token that an arc starts from, while
                # token_id is an ID of a token the arcs leads to.
                for token_id, head_id, deprel in zip(
                    sentence[ID],
                    sentence[HEAD],
                    sentence[DEPREL],
                    strict=True
                )
                if head_id is not None
            ]
        )
        result[UD_ARC_FROM] = ud_arcs_from
        result[UD_ARC_TO] = ud_arcs_to
        result[UD_DEPREL] = ud_deprels

    # Enhanced syntax.
    if DEPS in sentence and any(d is not None for d in sentence[DEPS]):
        eud_triples = [
            (
                id2idx[head_id] if head_id != ROOT_HEAD else id2idx[token_id],
                id2idx[token_id],
                deprel
            )
            for token_id, deps in zip(
                sentence[ID],
                sentence[DEPS],
                strict=True
            )
            if deps is not None
            for head_id, deprel in json.loads(deps).items()
        ]
        if eud_triples:
            eud_arcs_from, eud_arcs_to, eud_deprels = zip(*eud_triples)
            result[EUD_ARC_FROM] = eud_arcs_from
            result[EUD_ARC_TO] = eud_arcs_to
            result[EUD_DEPREL] = eud_deprels

    return result


def extract_unique_labels(dataset, column_name) -> list[str]:
    """Extract unique labels from a specific column in the dataset."""
    all_labels = itertools.chain.from_iterable(dataset[column_name])
    unique_labels = set(all_labels)
    unique_labels.discard(None)
    return unique_labels


def build_schema_with_class_labels(tagsets: dict[str, set]) -> Features:
    """Update the schema to use ClassLabel for specified columns."""

    # Updated features schema
    features = Features({
        SENT_ID: Value("string"),
        TEXT: Value("string"),
        WORD: Sequence(Value("string"))
    })

    if LEMMA_RULE in tagsets:
        # Sort to ensure consistent ordering of labels
        lemma_rule_tagset = sorted(tagsets[LEMMA_RULE])
        features[LEMMA_RULE] = Sequence(ClassLabel(names=lemma_rule_tagset))

    if JOINT_FEATS in tagsets:
        feats_tagset = sorted(tagsets[JOINT_FEATS])
        features[JOINT_FEATS] = Sequence(ClassLabel(names=feats_tagset))

    if UD_DEPREL in tagsets:
        features[UD_ARC_FROM] = Sequence(Value('int32'))
        features[UD_ARC_TO] = Sequence(Value('int32'))
        eud_deprels_tagset = sorted(tagsets[UD_DEPREL])
        features[UD_DEPREL] = Sequence(ClassLabel(names=eud_deprels_tagset))

    if EUD_DEPREL in tagsets:
        features[EUD_ARC_FROM] = Sequence(Value('int32'))
        features[EUD_ARC_TO] = Sequence(Value('int32'))
        eud_deprels_tagset = sorted(tagsets[EUD_DEPREL])
        features[EUD_DEPREL] = Sequence(ClassLabel(names=eud_deprels_tagset))

    if MISC in tagsets:
        misc_tagset = sorted(tagsets[MISC])
        features[MISC] = Sequence(ClassLabel(names=misc_tagset))

    if DEEPSLOT in tagsets:
        deepslot_tagset = sorted(tagsets[DEEPSLOT])
        features[DEEPSLOT] = Sequence(ClassLabel(names=deepslot_tagset))

    if SEMCLASS in tagsets:
        semclass_tagset = sorted(tagsets[SEMCLASS])
        features[SEMCLASS] = Sequence(ClassLabel(names=semclass_tagset))

    return features


def replace_none_with_ignore_index(example: dict, value: int = -100) -> dict:
    """
    Replace None labels with specified value.
    """
    assert value < 0
    for name, column in example.items():
        # Skip metadata fields (they are not lists).
        if isinstance(column, list):
            example[name] = [value if item is None else item for item in column]
    return example


def transform_dataset(dataset_dict: DatasetDict) -> Dataset:
    # Transform fields.
    dataset_column_names = {
        column
        for columns in dataset_dict.column_names.values()
        for column in columns
    }
    columns_to_remove = [ID, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS]

    # Remove range tokens and transform fields.
    dataset_dict = (
        dataset_dict
        .map(remove_range_tokens)
        .map(
            transform_fields,
            remove_columns=[
                column
                for column in columns_to_remove
                if column in dataset_column_names
            ]
        )
    )
    return dataset_dict


def collate_with_padding(batches: list[dict], padding_value: int = -100) -> dict:
    def gather_column(column_name: str) -> list:
        return [batch[column_name] for batch in batches]

    def stack_padded(column_name) -> LongTensor:
        return pad_sequences(gather_column(column_name), padding_value)

    def collate_syntax(arcs_from_name: str, arcs_to_name: str, deprel_name: str) -> LongTensor:
        batch_size = len(batches)
        arcs_counts = torch.tensor([len(batch[arcs_from_name]) for batch in batches])
        batch_idxs = torch.arange(batch_size).repeat_interleave(arcs_counts)
        from_idxs = torch.concat(gather_column(arcs_from_name))
        to_idxs = torch.concat(gather_column(arcs_to_name))
        deprels = torch.concat(gather_column(deprel_name))
        return torch.stack([batch_idxs, from_idxs, to_idxs, deprels], dim=1)

    def maybe_none(labels: LongTensor) -> LongTensor | None:
        return None if labels.max() == padding_value or labels.numel() == 0 else labels
    
    result = {
        "words": gather_column(WORD),
        "sent_ids": gather_column(SENT_ID),
        "texts": gather_column(TEXT)
    }

    columns = {column for batch in batches for column in batch}
    if LEMMA_RULE in columns:
        lemma_rules_batched = stack_padded(LEMMA_RULE)
        result["lemma_rules"] = maybe_none(lemma_rules_batched)

    if JOINT_FEATS in columns:
        joint_feats_batched = stack_padded(JOINT_FEATS)
        result["joint_feats"] = maybe_none(joint_feats_batched)

    if UD_DEPREL in columns:
        deps_ud_batched = collate_syntax(UD_ARC_FROM, UD_ARC_TO, UD_DEPREL)
        result["deps_ud"] = maybe_none(deps_ud_batched)

    if EUD_DEPREL in columns:
        deps_eud_batched = collate_syntax(EUD_ARC_FROM, EUD_ARC_TO, EUD_DEPREL)
        result["deps_eud"] = maybe_none(deps_eud_batched)

    if MISC in columns:
        miscs_batched = stack_padded(MISC)
        result["miscs"] = maybe_none(miscs_batched)

    if DEEPSLOT in columns:
        deepslots_batched = stack_padded(DEEPSLOT)
        result["deepslots"] = maybe_none(deepslots_batched)

    if SEMCLASS in columns:
        semclasses_batched = stack_padded(SEMCLASS)
        result["semclasses"] = maybe_none(semclasses_batched)

    return result