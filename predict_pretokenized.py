import argparse

from transformers import pipeline


def parse_conllu_to_token_lists(filepath):
    def is_range_id(x: str) -> bool:
        try:
            a, b = x.split('-')
            return a.isdecimal() and b.isdecimal()
        except ValueError:
            return False

    sentences = []
    current_tokens = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # If it's a comment line starting with '# text =', start a new sentence
            if line.startswith('# text ='):
                if current_tokens:
                    sentences.append(current_tokens)
                    current_tokens = []

            # Skip empty lines
            elif not line:
                continue

            # Process token lines
            elif not line.startswith('#'):
                columns = line.split('\t')
                assert 2 <= len(columns)
                token_id = columns[0]
                word = columns[1]
                if is_range_id(token_id):
                    # Skip range tokens
                    continue
                current_tokens.append(word)

    # Add the last sentence if any
    if current_tokens:
        sentences.append(current_tokens)

    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference model on pre-tokenized conllu texts."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Conllu file there tokenized sentences are taken from"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output conllu file there annotated sentences are written to"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Cobald parser model to inference"
    )
    args = parser.parse_args()

    # HACK (How does a mathematician boil water?)
    # Pipeline takes text as input, splits it to sentences and words.
    # What if we want to feed him pre-tokenized sentences?
    # Reduce them to a previously solved problem! concatenate tokens
    # into sentences using unique symbol, then split them upon this
    # delimiter in tokenizer.
    token_delimiter = '\t'
    dummy_tokenizer = lambda sentence: sentence.split(token_delimiter)
    dummy_sentenizer = lambda sentence: [sentence]

    pipe = pipeline(
        "token-classification",
        model=args.model,
        tokenizer=dummy_tokenizer,
        sentenizer=dummy_sentenizer,
        trust_remote_code=True
    )

    batch_size = 64
    sentences = []
    for tokens in parse_conllu_to_token_lists(args.input):
        assert token_delimiter not in tokens
        sentences.append(token_delimiter.join(tokens))

    outputs = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        outputs.extend(pipe(batch_sentences, output_format='str', batch_size=batch_size))

    with open(args.output, 'w') as f:
        f.write('\n\n'.join(outputs))
