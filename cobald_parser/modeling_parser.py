import torch
from torch import nn
from torch import LongTensor
from transformers import PreTrainedModel

from .configuration import CobaldParserConfig
from .encoder import WordTransformerEncoder
from .mlp_classifier import MlpClassifier
from .dependency_classifier import DependencyClassifier
from .utils import build_padding_mask


ELLIPSIS_TOKEN = "_"


def _build_ellipsis_mask(sentences: list[list[str]], device) -> torch.Tensor:
    """Build a boolean mask where True = non-ellipsis token, False = ellipsis token."""
    masks = [
        torch.tensor(
            [word is not None and word != ELLIPSIS_TOKEN for word in sentence],
            dtype=torch.bool, device=device
        )
        for sentence in sentences
    ]
    return torch.nn.utils.rnn.pad_sequence(masks, padding_value=False, batch_first=True)


def _replace_ellipsis(sentences: list[list[str]], mask_token: str) -> list[list[str]]:
    """Replace ellipsis tokens with the given mask token."""
    return [
        [mask_token if (word is None or word == ELLIPSIS_TOKEN) else word for word in sentence]
        for sentence in sentences
    ]


class CobaldParser(PreTrainedModel):
    """Morpho-Syntax-Semantic Parser."""

    config_class = CobaldParserConfig

    def __init__(self, config: CobaldParserConfig):
        super().__init__(config)

        self.encoder = WordTransformerEncoder(
            model_name=config.encoder_model_name
        )
        embedding_size = self.encoder.get_embedding_size()

        self.classifiers = nn.ModuleDict()
        if "lemma_rule" in config.vocabulary:
            self.classifiers["lemma_rule"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.lemma_classifier_hidden_size,
                n_classes=len(config.vocabulary["lemma_rule"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "joint_feats" in config.vocabulary:
            self.classifiers["joint_feats"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.morphology_classifier_hidden_size,
                n_classes=len(config.vocabulary["joint_feats"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "ud_deprel" in config.vocabulary or "eud_deprel" in config.vocabulary:
            self.classifiers["syntax"] = DependencyClassifier(
                input_size=embedding_size,
                hidden_size=config.dependency_classifier_hidden_size,
                n_rels_ud=len(config.vocabulary.get("ud_deprel", {})),
                n_rels_eud=len(config.vocabulary.get("eud_deprel", {})),
                activation=config.activation,
                dropout=config.dropout
            )
        if "misc" in config.vocabulary:
            self.classifiers["misc"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.misc_classifier_hidden_size,
                n_classes=len(config.vocabulary["misc"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "deepslot" in config.vocabulary:
            self.classifiers["deepslot"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.deepslot_classifier_hidden_size,
                n_classes=len(config.vocabulary["deepslot"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "semclass" in config.vocabulary:
            self.classifiers["semclass"] = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.semclass_classifier_hidden_size,
                n_classes=len(config.vocabulary["semclass"]),
                activation=config.activation,
                dropout=config.dropout
            )

    def forward(
        self,
        words: list[list[str]],
        lemma_rules: LongTensor = None,
        joint_feats: LongTensor = None,
        deps_ud: LongTensor = None,
        deps_eud: LongTensor = None,
        miscs: LongTensor = None,
        deepslots: LongTensor = None,
        semclasses: LongTensor = None,
        sent_ids: list[str] = None,
        texts: list[str] = None,
        inference_mode: bool = False
    ) -> dict:
        output = {}
        output["words"] = words

        # Build ellipsis mask: True for real tokens, False for ellipsis.
        ellipsis_mask = _build_ellipsis_mask(words, self.device)

        # Replace ellipsis tokens with the encoder's mask token before encoding.
        words_for_encoder = _replace_ellipsis(words, self.config.ellipsis_mask_token)

        # Encode all words (ellipsis positions get mask-token embeddings).
        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(words_for_encoder)

        output["loss"] = 0.0

        # Predict lemmas and morphological features (exclude ellipsis from loss).
        if "lemma_rule" in self.classifiers:
            lemma_output = self.classifiers["lemma_rule"](embeddings, lemma_rules, mask=ellipsis_mask)
            output["lemma_rules"] = lemma_output['preds']
            output["loss"] += lemma_output['loss']

        if "joint_feats" in self.classifiers:
            joint_feats_output = self.classifiers["joint_feats"](embeddings, joint_feats, mask=ellipsis_mask)
            output["joint_feats"] = joint_feats_output['preds']
            output["loss"] += joint_feats_output['loss']

        # Predict syntax (ellipsis tokens participate normally).
        if "syntax" in self.classifiers:
            padding_mask = build_padding_mask(words, self.device)
            # No null masking — all tokens (including ellipsis) participate in syntax.
            null_mask = torch.ones_like(padding_mask)
            deps_output = self.classifiers["syntax"](
                embeddings,
                deps_ud,
                deps_eud,
                null_mask,
                padding_mask
            )
            if 'preds_ud' in deps_output:
                output["deps_ud"] = deps_output['preds_ud']
                output["loss"] += deps_output['loss_ud']
            if 'preds_eud' in deps_output:
                output["deps_eud"] = deps_output['preds_eud']
                output["loss"] += deps_output['loss_eud']

        # Predict miscellaneous features.
        if "misc" in self.classifiers:
            misc_output = self.classifiers["misc"](embeddings, miscs)
            output["miscs"] = misc_output['preds']
            output["loss"] += misc_output['loss']

        # Predict semantics.
        if "deepslot" in self.classifiers:
            deepslot_output = self.classifiers["deepslot"](embeddings, deepslots)
            output["deepslots"] = deepslot_output['preds']
            output["loss"] += deepslot_output['loss']

        if "semclass" in self.classifiers:
            semclass_output = self.classifiers["semclass"](embeddings, semclasses)
            output["semclasses"] = semclass_output['preds']
            output["loss"] += semclass_output['loss']

        return output
