from transformers import PretrainedConfig


class CobaldParserConfig(PretrainedConfig):
    model_type = "cobald_parser"

    def __init__(
        self,
        encoder_model_name: str = None,
        ellipsis_mask_token: str = "[MASK]",
        lemma_classifier_hidden_size: int = 0,
        morphology_classifier_hidden_size: int = 0,
        dependency_classifier_hidden_size: int = 0,
        misc_classifier_hidden_size: int = 0,
        deepslot_classifier_hidden_size: int = 0,
        semclass_classifier_hidden_size: int = 0,
        activation: str = 'relu',
        dropout: float = 0.1,
        vocabulary: dict[dict[int, str]] = {},
        **kwargs
    ):
        self.encoder_model_name = encoder_model_name
        self.ellipsis_mask_token = ellipsis_mask_token
        self.lemma_classifier_hidden_size = lemma_classifier_hidden_size
        self.morphology_classifier_hidden_size = morphology_classifier_hidden_size
        self.dependency_classifier_hidden_size = dependency_classifier_hidden_size
        self.misc_classifier_hidden_size = misc_classifier_hidden_size
        self.deepslot_classifier_hidden_size = deepslot_classifier_hidden_size
        self.semclass_classifier_hidden_size = semclass_classifier_hidden_size
        self.activation = activation
        self.dropout = dropout
        # The serialized config stores mappings as strings,
        # e.g. {"0": "acl", "1": "conj"}, so we have to convert them to int.
        self.vocabulary = {
            column: {int(k): v for k, v in labels.items()}
            for column, labels in vocabulary.items()
        }
        super().__init__(**kwargs)