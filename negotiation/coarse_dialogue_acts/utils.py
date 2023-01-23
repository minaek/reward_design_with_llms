from nltk.tokenize import word_tokenize


def tokenize(sentence: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    return tokens
