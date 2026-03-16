import json
import os
from collections import defaultdict

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import (
    HfArgumentParser,
    TrainingArguments
)

from cobald_parser import (
    CobaldParserConfig,
    CobaldParser,
    ConlluTokenClassificationPipeline
)
from src.processing import (
    transform_dataset,
    extract_unique_labels,
    build_schema_with_class_labels,
    replace_none_with_ignore_index,
    collate_with_padding,
    LEMMA_RULE,
    JOINT_FEATS,
    UD_DEPREL,
    EUD_DEPREL,
    MISC,
    SEMCLASS,
    DEEPSLOT,
)
from src.callbacks import GradualUnfreezeCallback
from src.trainer import CustomTrainer
from src.metrics import compute_metrics

# Standard CoNLL-U columns (by index).
CONLLU_COLUMNS = ["id", "word", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
# Optional extra columns used by CoBaLD.
CONLLU_EXTRA_COLUMNS = ["deepslot", "semclass"]


def parse_conllu(filepath: str) -> Dataset:
    """Parse a CoNLL-U file into a HuggingFace Dataset (one row per sentence)."""
    sentences = []
    current = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                current = {"sent_id": line.split("=", 1)[1].strip()}
            elif line.startswith("# text"):
                if current is None:
                    current = {}
                current["text"] = line.split("=", 1)[1].strip()
            elif line.startswith("#"):
                continue
            elif line == "":
                if current is not None:
                    sentences.append(current)
                    current = None
            else:
                if current is None:
                    current = {}
                fields = line.split("\t")
                columns = CONLLU_COLUMNS + CONLLU_EXTRA_COLUMNS[:max(0, len(fields) - len(CONLLU_COLUMNS))]
                # Columns where "_" means missing value (not literal underscore).
                nullable_columns = {"lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc", "deepslot", "semclass"}
                for col, val in zip(columns, fields):
                    current.setdefault(col, [])
                    if val == "_" and col in nullable_columns:
                        current[col].append(None)
                    elif col == "head" and val != "_":
                        current[col].append(int(val))
                    elif col == "deps" and val != "_":
                        # Convert "head1:deprel1|head2:deprel2" to JSON string.
                        pairs = {}
                        for pair in val.split("|"):
                            h, d = pair.split(":", 1)
                            pairs[h] = d
                        current[col].append(json.dumps(pairs))
                    else:
                        current[col].append(val)

    if current is not None:
        sentences.append(current)

    # Collect all column names across sentences.
    all_columns = dict.fromkeys(
        col for sent in sentences for col in sent if col not in ("sent_id", "text")
    )
    # Ensure every sentence has all columns.
    for sent in sentences:
        for col in all_columns:
            sent.setdefault(col, [])

    return Dataset.from_list(sentences)


def load_conllu_folder(data_dir: str) -> DatasetDict:
    """Load train.conllu, dev.conllu and test.conllu from a folder."""
    splits = {}
    file_to_split = {"train.conllu": "train", "dev.conllu": "validation", "test.conllu": "test"}
    for filename, split_name in file_to_split.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            splits[split_name] = parse_conllu(path)
    if not splits:
        raise FileNotFoundError(f"No .conllu files found in {data_dir}")
    return DatasetDict(splits)


def parse_datasets(value: str) -> list[tuple]:
    result = []
    datasets_configs = value.split(',')
    for dataset_config in datasets_configs:
        parts = dataset_config.split(':')
        if len(parts) != 2:
            raise ValueError(f"Dataset '{value}' is not in the format 'name:config'")
        dataset, config = parts
        result.append((dataset, config))
    return result


def build_shared_tagsets(datasets_configs: list[tuple], allowed_columns: set = None) -> dict:
    tagsets = defaultdict(set)
    for dataset_name, config_name in datasets_configs:
        external_dataset_dict = load_dataset(dataset_name, name=config_name)
        external_dataset_dict = transform_dataset(external_dataset_dict)
        external_dataset = concatenate_datasets(external_dataset_dict.values())
        for column_name in external_dataset.column_names:
            # Skip columns that are not marked as allowed
            if allowed_columns is not None and column_name not in allowed_columns:
                continue
            tagsets[column_name] |= extract_unique_labels(external_dataset, column_name)
    return tagsets


def update_vocabulary(config, features):
    for column in [LEMMA_RULE, JOINT_FEATS, UD_DEPREL, EUD_DEPREL, MISC, DEEPSLOT, SEMCLASS]:
        if column in features and hasattr(features[column], 'feature') and hasattr(features[column].feature, 'names'):
            labels = features[column].feature.names
            config.vocabulary[column] = dict(enumerate(labels))


def transfer_pretrained(model, pretrained_model):
    if not isinstance(pretrained_model, CobaldParser):
        raise ValueError(f"Pretrained model must be CobaldParser class instance")

    # Transfer encoder
    model.encoder = pretrained_model.encoder

    a = set(model.config.vocabulary[EUD_DEPREL].items())
    b = set(pretrained_model.config.vocabulary[EUD_DEPREL].items())

    print(f"diff: {a - b}")

    # Transfer classifiers
    for name in model.classifiers:
        if name in pretrained_model.classifiers:
            try:
                # Try to transfer weights from pretrained classifier if it matches
                # the shape of the model's classifier (e.g. hidden_size, n_classes, etc.)
                pretrained_classifier_state = pretrained_model.classifiers[name].state_dict()
                model.classifiers[name].load_state_dict(pretrained_classifier_state)
                print(f"Successfuly transfered {name} classifier")
            except Exception as e:
                print(f"Could not transfer {name} classifier:\n{e}")


if __name__ == "__main__":
    # Use HfArgumentParser with the built-in TrainingArguments class
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('--model_config', required=True)
    parser.add_argument(
        '--data_dir', required=True,
        help="Path to folder with train.conllu, dev.conllu and test.conllu"
    )
    parser.add_argument(
        '--external_datasets',
        type=parse_datasets,
        nargs="?",
        help="External datasets whose colums will be added to the model's vocabulary "
        "in format `dataset1_path:config1_name,dataset2_path:config2_name,...` "
        "(use `default` if no other configs exist). "
        "Example: `CoBaLD/enhanced-cobald:en,CoBaLD/enhanced-ud-syntax:default`."
    )
    parser.add_argument('--finetune_from')
    parser.add_argument(
        '--columns',
        nargs='+',
        help="CoNLL-U columns to keep for training (e.g. upos feats head deprel). "
        "Columns 'id' and 'word' are always kept."
    )

    # Parse command-line arguments.
    training_args, custom_args = parser.parse_args_into_dataclasses()

    target_dataset_dict = load_conllu_folder(custom_args.data_dir)

    # Remove columns not requested by the user.
    if custom_args.columns:
        keep = {"id", "word", "sent_id", "text"} | set(custom_args.columns)
        drop = [c for c in target_dataset_dict["train"].column_names if c not in keep]
        if drop:
            target_dataset_dict = target_dataset_dict.remove_columns(drop)
    target_dataset_dict = transform_dataset(target_dataset_dict)

    # Build tagsets from the local dataset.
    allowed_columns = target_dataset_dict['train'].column_names
    target_all = concatenate_datasets(target_dataset_dict.values())
    tagsets = defaultdict(set)
    for column_name in target_all.column_names:
        if column_name in allowed_columns:
            tagsets[column_name] |= extract_unique_labels(target_all, column_name)

    # Extend tagsets with external HF datasets (if any).
    if custom_args.external_datasets:
        external_tagsets = build_shared_tagsets(
            custom_args.external_datasets,
            allowed_columns=allowed_columns
        )
        for col, labels in external_tagsets.items():
            tagsets[col] |= labels

    schema = build_schema_with_class_labels(tagsets)

    # Final processing.
    target_dataset_dict = (
        target_dataset_dict
        .cast(schema)
        .map(replace_none_with_ignore_index)
        .with_format("torch")
    )

    # Create and configure model.
    model_config = CobaldParserConfig.from_json_file(custom_args.model_config)
    # Load vocabulary into config (as it must be saved along the model).
    update_vocabulary(model_config, target_dataset_dict['train'].features)

    # Manually set some parameters for this specific workflow to work.
    training_args.remove_unused_columns = False
    training_args.label_names = []
    for dataset_column, parser_input in (
        (LEMMA_RULE, "lemma_rules"),
        (JOINT_FEATS, "joint_feats"),
        (UD_DEPREL, "deps_ud"),
        (EUD_DEPREL, "deps_eud"),
        (MISC, "miscs"),
        (DEEPSLOT, "deepslots"),
        (SEMCLASS, "semclasses")
    ):
        if dataset_column in model_config.vocabulary:
            training_args.label_names.append(parser_input)

    model = CobaldParser(model_config)

    if custom_args.finetune_from:
        pretrained_model = CobaldParser.from_pretrained(
            custom_args.finetune_from,
            trust_remote_code=True
        )
        transfer_pretrained(model, pretrained_model)

    # Create trainer and train the model.
    unfreeze_callback = GradualUnfreezeCallback()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset_dict['train'],
        eval_dataset=target_dataset_dict['validation'],
        data_collator=collate_with_padding,
        # Wth? See notes at compute_metrics.
        compute_metrics=lambda x: compute_metrics(x, training_args.label_names),
        callbacks=[unfreeze_callback]
    )
    trainer.train(ignore_keys_for_eval=["words", "sent_ids", "texts"])

    # Save and push model to hub (if push_to_hub is set).
    trainer.save_model()

    if training_args.hub_model_id:
        pipe = ConlluTokenClassificationPipeline(model)
        pipe.push_to_hub(training_args.hub_model_id)