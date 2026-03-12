from torch import nn
from torch import LongTensor
from transformers import PreTrainedModel

from .configuration import CobaldParserConfig
from .encoder import WordTransformerEncoder
from .mlp_classifier import MlpClassifier
from .dependency_classifier import DependencyClassifier
from .utils import (
    build_padding_mask,
    build_null_mask,
    prepend_cls,
    remove_nulls,
    add_nulls
)


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
        self.classifiers["null"] = MlpClassifier(
            input_size=self.encoder.get_embedding_size(),
            hidden_size=config.null_classifier_hidden_size,
            n_classes=config.consecutive_null_limit + 1,
            activation=config.activation,
            dropout=config.dropout
        )
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
        counting_masks: LongTensor = None,
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

        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        words_with_cls = prepend_cls(words)
        words_without_nulls = remove_nulls(words_with_cls)
        # Embeddings of words without nulls.
        embeddings_without_nulls = self.encoder(words_without_nulls)
        # Predict nulls.
        null_output = self.classifiers["null"](embeddings_without_nulls, counting_masks)
        output["counting_mask"] = null_output['preds']
        output["loss"] = null_output["loss"]

        # "Teacher forcing": during training, pass the original words (with gold nulls)
        # to the classification heads, so that they are trained upon correct sentences.
        if inference_mode:
            # Restore predicted nulls in the original sentences.
            output["words"] = add_nulls(words, null_output["preds"])
        else:
            output["words"] = words

        # Encode words with nulls.
        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(output["words"])

        # Predict lemmas and morphological features.
        if "lemma_rule" in self.classifiers:
            lemma_output = self.classifiers["lemma_rule"](embeddings, lemma_rules)
            output["lemma_rules"] = lemma_output['preds']
            output["loss"] += lemma_output['loss']

        if "joint_feats" in self.classifiers:
            joint_feats_output = self.classifiers["joint_feats"](embeddings, joint_feats)
            output["joint_feats"] = joint_feats_output['preds']
            output["loss"] += joint_feats_output['loss']

        # Predict syntax.
        if "syntax" in self.classifiers:
            padding_mask = build_padding_mask(output["words"], self.device)
            null_mask = build_null_mask(output["words"], self.device)
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