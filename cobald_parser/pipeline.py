from typing import override

from transformers import Pipeline

from src.lemmatize_helper import reconstruct_lemma
from cobald_parser.modeling_parser import ELLIPSIS_TOKEN


class ConlluTokenClassificationPipeline(Pipeline):
    def __init__(
        self,
        model,
        tokenizer: callable = None,
        sentenizer: callable = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.tokenizer = tokenizer
        self.sentenizer = sentenizer

    @override
    def _sanitize_parameters(self, output_format: str = 'list', **kwargs):
        if output_format not in ['list', 'str']:
            raise ValueError(
                f"output_format must be 'str' or 'list', not {output_format}"
            )
        # capture output_format for postprocessing
        return {}, {}, {'output_format': output_format}

    @override
    def preprocess(self, inputs: str) -> dict:
        if not isinstance(inputs, str):
            raise ValueError("pipeline input must be string (text)")

        sentences = [sentence for sentence in self.sentenizer(inputs)]
        words = [
            [word for word in self.tokenizer(sentence)]
            for sentence in sentences
        ]
        # stash for later post‐processing
        self._texts = sentences
        return {"words": words}

    @override
    def _forward(self, model_inputs: dict) -> dict:
        return self.model(**model_inputs, inference_mode=True)

    @override
    def postprocess(self, model_outputs: dict, output_format: str) -> list[dict] | str:
        sentences = self._decode_model_output(model_outputs)
        # Format sentences into CoNLL-U string if requested.
        if output_format == 'str':
            sentences = self._format_as_conllu(sentences)
        return sentences

    def _decode_model_output(self, model_outputs: dict) -> list[dict]:
        n_sentences = len(model_outputs["words"])

        sentences_decoded = []
        for i in range(n_sentences):

            def select_arcs(arcs, batch_idx):
                # Select arcs where batch index == batch_idx
                # Return tensor of shape [n_selected_arcs, 3]
                return arcs[arcs[:, 0] == batch_idx][:, 1:]
            
            # Model outputs are padded tensors, so only leave first `n_words` labels.
            n_words = len(model_outputs["words"][i])

            optional_tags = {}
            if "lemma_rules" in model_outputs:
                optional_tags["lemma_rule_ids"] = model_outputs["lemma_rules"][i, :n_words].tolist()
            if "joint_feats" in model_outputs:
                optional_tags["joint_feats_ids"] = model_outputs["joint_feats"][i, :n_words].tolist()
            if "deps_ud" in model_outputs:
                optional_tags["deps_ud"] = select_arcs(model_outputs["deps_ud"], i).tolist()
            if "deps_eud" in model_outputs:
                optional_tags["deps_eud"] = select_arcs(model_outputs["deps_eud"], i).tolist()
            if "miscs" in model_outputs:
                optional_tags["misc_ids"] = model_outputs["miscs"][i, :n_words].tolist()
            if "deepslots" in model_outputs:
                optional_tags["deepslot_ids"] = model_outputs["deepslots"][i, :n_words].tolist()
            if "semclasses" in model_outputs:
                optional_tags["semclass_ids"] = model_outputs["semclasses"][i, :n_words].tolist()

            sentence_decoded = self._decode_sentence(
                text=self._texts[i],
                words=model_outputs["words"][i],
                **optional_tags,
            )
            sentences_decoded.append(sentence_decoded)
        return sentences_decoded

    def _decode_sentence(
        self,
        text: str,
        words: list[str],
        lemma_rule_ids: list[int] = None,
        joint_feats_ids: list[int] = None,
        deps_ud: list[list[int]] = None,
        deps_eud: list[list[int]] = None,
        misc_ids: list[int] = None,
        deepslot_ids: list[int] = None,
        semclass_ids: list[int] = None
    ) -> dict:

        # Enumerate words in the sentence, starting from 1.
        ids = self._enumerate_words(words)

        result = {
            "text": text,
            "words": words,
            "ids": ids
        }

        # Build ellipsis flags for this sentence.
        is_ellipsis = [w is None or w == ELLIPSIS_TOKEN for w in words]

        # Decode lemmas (skip ellipsis).
        if lemma_rule_ids:
            result["lemmas"] = [
                "_" if is_ellipsis[i] else reconstruct_lemma(
                    word,
                    self.model.config.vocabulary["lemma_rule"][lemma_rule_id]
                )
                for i, (word, lemma_rule_id) in enumerate(zip(words, lemma_rule_ids, strict=True))
            ]
        # Decode POS and features (skip ellipsis).
        if joint_feats_ids:
            upos, xpos, feats = zip(
                *[
                    ("_", "_", "_") if is_ellipsis[i] else
                    self.model.config.vocabulary["joint_feats"][joint_feats_id].split('#')
                    for i, joint_feats_id in enumerate(joint_feats_ids)
                ],
                strict=True
            )
            result["upos"] = list(upos)
            result["xpos"] = list(xpos)
            result["feats"] = list(feats)
        # Decode syntax.
        renumerate_and_decode_arcs = lambda arcs, id2rel: [
            (
                # ids stores inverse mapping from internal numeration to the standard
                # conllu numeration, so simply use ids[internal_idx] to retrieve token id
                # from internal index.
                ids[arc_from] if arc_from != arc_to else '0',
                ids[arc_to],
                id2rel[deprel_id]
            )
            for arc_from, arc_to, deprel_id in arcs
        ]
        if deps_ud:
            result["deps_ud"] = renumerate_and_decode_arcs(
                deps_ud,
                self.model.config.vocabulary["ud_deprel"]
            )
        if deps_eud:
            result["deps_eud"] = renumerate_and_decode_arcs(
                deps_eud,
                self.model.config.vocabulary["eud_deprel"]
            )
        # Decode misc.
        if misc_ids:
            result["miscs"] = [
                self.model.config.vocabulary["misc"][misc_id]
                for misc_id in misc_ids
            ]
        # Decode semantics.
        if deepslot_ids:
            result["deepslots"] = [
                self.model.config.vocabulary["deepslot"][deepslot_id] 
                for deepslot_id in deepslot_ids
            ]
        if semclass_ids:
            result["semclasses"] = [
                self.model.config.vocabulary["semclass"][semclass_id] 
                for semclass_id in semclass_ids
            ]
        return result

    @staticmethod
    def _enumerate_words(words: list[str]) -> list[str]:
        return [f"{i + 1}" for i in range(len(words))]

    @staticmethod
    def _format_as_conllu(sentences: list[dict]) -> str:
        """
        Format a list of sentence dicts into a CoNLL-U formatted string.
        """
        formatted = []
        for sentence in sentences:
            # The first line is a text matadata.
            lines = [f"# text = {sentence['text']}"]

            id2idx = {token_id: idx for idx, token_id in enumerate(sentence['ids'])}

            # Basic syntax.
            heads = [''] * len(id2idx)
            deprels = [''] * len(id2idx)
            if "deps_ud" in sentence:
                for arc_from, arc_to, deprel in sentence['deps_ud']:
                    token_idx = id2idx[arc_to]
                    heads[token_idx] = arc_from
                    deprels[token_idx] = deprel

            # Enhanced syntax.
            deps_dicts = [{} for _ in range(len(id2idx))]
            if "deps_eud" in sentence:
                for arc_from, arc_to, deprel in sentence['deps_eud']:
                    token_idx = id2idx[arc_to]
                    deps_dicts[token_idx][arc_from] = deprel

            for idx, token_id in enumerate(sentence['ids']):
                word = sentence['words'][idx]
                lemma = sentence.get('lemmas', ['_'] * len(sentence['ids']))[idx]
                upos = sentence.get('upos', ['_'] * len(sentence['ids']))[idx]
                xpos = sentence.get('xpos', ['_'] * len(sentence['ids']))[idx]
                feats = sentence.get('feats', ['_'] * len(sentence['ids']))[idx]
                head = heads[idx] or '_'
                deprel = deprels[idx] or '_'
                deps = '|'.join(f"{h}:{r}" for h, r in deps_dicts[idx].items()) or '_'
                misc = sentence.get('miscs', ['_'] * len(sentence['ids']))[idx]
                # Standard 10-column CoNLL-U format
                line = '\t'.join([
                    token_id, word, lemma, upos, xpos, feats, head,
                    deprel, deps, misc
                ])
                lines.append(line)
            formatted.append('\n'.join(lines))
        return '\n\n'.join(formatted)