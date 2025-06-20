import re
import string
import subprocess
import sys
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import os

tqdm.pandas()
per_target = 7000
sample_df = pd.read_parquet(f"data/sample_{per_target}_per_target.parquet", engine='pyarrow')

text_column = 'text'

if not os.path.exists("data/nlps.pkl"):
    def install_spacy_model():
        english_model = "en_core_web_sm"
        try:
            import spacy
            nlp = spacy.load(english_model)
            print("spaCy model already installed")
            return nlp
        except (ImportError, OSError):
            print("Installing spaCy English model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", english_model])
            import spacy
            return spacy.load(english_model)

    nlp = install_spacy_model()
    print("Processing text with spaCy...")
    nlps = sample_df[text_column].progress_apply(lambda x: nlp(x) if isinstance(x, str) else None).tolist()
    joblib.dump(nlps, "data/nlps.pkl")
else:
    print("Loading preprocessed text from joblib...")
    nlps = joblib.load("data/nlps.pkl")
    print(f"Loaded {len(nlps)} preprocessed texts from joblib.")

sample_df['uniq_words_ratio'] = sample_df[text_column].apply(
    lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)


def word_length_stats(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0, 0.0

    words = text.split()
    cleaned_words = []

    for word in words:
        cleaned_word = word.strip(string.punctuation)
        if cleaned_word:
            cleaned_words.append(cleaned_word)

    if not cleaned_words:
        return 0.0, 0.0

    word_lengths = [len(word) for word in cleaned_words]

    mean_length = np.mean(word_lengths)
    variance_length = np.var(word_lengths)

    return mean_length, variance_length


def apply_word_length_stats(df, text_column):
    word_stats = df[text_column].apply(word_length_stats)

    df['mean_word_length'] = word_stats.apply(lambda x: x[0])
    df['word_length_variance'] = word_stats.apply(lambda x: x[1])

    return df


def function_word_frequency(text):
    function_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'shall', 'not', 'no', 'yes', 'if', 'when', 'where', 'why', 'how', 'what',
        'who', 'which', 'whose', 'whom', 'while', 'since', 'because', 'although', 'though',
        'as', 'so', 'than', 'such', 'both', 'either', 'neither', 'each', 'every', 'all',
        'any', 'some', 'many', 'much', 'few', 'little', 'more', 'most', 'less', 'least'
    }

    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    words = text.lower().split()
    cleaned_words = []

    for word in words:
        cleaned_word = word.strip(string.punctuation)
        if cleaned_word:
            cleaned_words.append(cleaned_word)

    if not cleaned_words:
        return 0.0

    function_word_count = sum(1 for word in cleaned_words if word in function_words)

    return function_word_count / len(cleaned_words)


def punctuation_density_diversity(text):
    if not text or not isinstance(text, str):
        return 0.0, 0.0, 0, 0, 0

    punctuation_chars = [char for char in text if char in string.punctuation]
    total_chars = len(text)
    punct_count = len(punctuation_chars)

    if punct_count == 0:
        return 0.0, 0.0, total_chars, 0, 0

    density = punct_count / total_chars if total_chars > 0 else 0.0

    unique_punct = set(punctuation_chars)
    unique_punct_count = len(unique_punct)
    diversity = unique_punct_count / punct_count if punct_count > 0 else 0.0

    return density, diversity, total_chars, punct_count, unique_punct_count


def count_grammar_issues_spacy(row):
    doc = nlps[row.name]
    issues = 0

    if len(doc) == 0:
        return 0.0

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            if token.tag_ in ["NNS", "NNPS"] and token.head.tag_ in ["VBZ"]:  # plural subject, singular verb
                issues += 1
            elif token.tag_ in ["NN", "NNP"] and token.head.tag_ in ["VBP"]:  # singular subject, plural verb
                issues += 1

        if token.pos_ == "VERB" and token.tag_ == "VBG" and token.dep_ == "ROOT":
            if not any(child.pos_ == "AUX" for child in token.children):
                issues += 1

        if token.pos_ == "NOUN" and token.i > 0:
            prev_token = doc[token.i - 1]
            if prev_token.pos_ not in ["DET", "PRON"] and token.tag_ in ["NN"]:
                if not token.is_title and token.text.lower() not in ["time", "money", "water", "information"]:
                    issues += 0.5  # Soft penalty as this is complex

    return (issues / len(doc)) * 100


def sentence_length_variance(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0, 0.0

    sentences = re.split(r'[.!?]+', text)

    sentence_lengths = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            words = sentence.split()
            if words:
                sentence_lengths.append(len(words))

    if len(sentence_lengths) < 2:
        return 0.0, 0.0

    return np.var(sentence_lengths), np.mean(sentence_lengths)


def hedging_language_frequency(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    hedging_words = {
        'perhaps', 'maybe', 'possibly', 'probably', 'likely', 'unlikely',
        'might', 'may', 'could', 'would', 'should', 'seem', 'seems',
        'appear', 'appears', 'suggest', 'suggests', 'indicate', 'indicates',
        'tend', 'tends', 'somewhat', 'rather', 'quite', 'fairly',
        'relatively', 'generally', 'usually', 'typically', 'often',
        'sometimes', 'occasionally', 'potentially', 'presumably',
        'apparently', 'evidently', 'arguably', 'conceivably',
        'theoretically', 'hypothetically', 'arguably', 'supposedly',
        'allegedly', 'reportedly', 'roughly', 'approximately',
        'around', 'about', 'nearly', 'almost', 'sort of', 'kind of'
    }

    words = text.lower().split()
    if not words:
        return 0.0

    hedging_count = sum(1 for word in words if word.strip('.,!?;:') in hedging_words)

    text_lower = text.lower()
    hedging_phrases = ['sort of', 'kind of', 'it seems', 'it appears', 'i think', 'i believe']
    for phrase in hedging_phrases:
        hedging_count += text_lower.count(phrase)

    return (hedging_count / len(words)) * 100


def count_syllables(word):
    word = word.lower().strip()
    if not word:
        return 0

    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0

    vowels = 'aeiouy'
    syllable_count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel

    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1

    return max(1, syllable_count)


def average_syllables_per_word(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    total_syllables = sum(count_syllables(word) for word in words)

    return total_syllables / len(words)


def exclamation_frequency(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    exclamation_count = text.count('!')

    return (exclamation_count / len(words)) * 100


def word_length_variance(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    words = re.findall(r'\b\w+\b', text.lower())

    if len(words) < 2:
        return 0.0

    word_lengths = [len(word) for word in words]

    return np.var(word_lengths)


def subordinate_clause_ratio(row):
    doc = nlps[row.name]
    total_clauses = 0
    subordinate_clauses = 0

    for token in doc:
        if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl']:
            subordinate_clauses += 1
        if token.dep_ in ['ROOT', 'ccomp', 'xcomp', 'advcl', 'acl', 'relcl']:
            total_clauses += 1

    return (subordinate_clauses / total_clauses * 100) if total_clauses > 0 else 0


def modal_verb_patterns(row):
    doc = nlps[row.name]
    modals = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']

    modal_counts = Counter()

    for token in doc:
        if token.lemma_.lower() in modals:
            modal_counts[token.lemma_.lower()] += 1

    unique_modals = len(modal_counts)

    return (unique_modals / len(modals) * 100) if modals else 0


def transition_word_overuse(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    transition_words = {
        'first', 'second', 'third', 'finally', 'additionally', 'furthermore',
        'moreover', 'however', 'therefore', 'consequently', 'nevertheless',
        'meanwhile', 'subsequently', 'alternatively', 'specifically',
        'particularly', 'especially', 'notably', 'importantly', 'significantly'
    }

    sentences = re.split(r'[.!?]+', text)
    if len(sentences) < 2:
        return 0.0

    transition_starts = 0
    valid_sentences = 0

    for sent in sentences:
        sent = sent.strip()
        if sent:
            words = sent.split()
            if words:
                valid_sentences += 1
                first_word = words[0].lower().strip(string.punctuation)
                if first_word in transition_words:
                    transition_starts += 1

    return (transition_starts / valid_sentences) * 100 if valid_sentences > 0 else 0.0

def rare_word_frequency(row):
    doc = nlps[row.name]

    common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he',
                    'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
                    'about', 'who', 'get', 'which', 'go', 'me'}

    content_words = [token.lemma_.lower() for token in doc
                     if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']
                     and not token.is_stop and token.is_alpha]

    if len(content_words) == 0:
        return 0

    rare_words = [word for word in content_words if word not in common_words and len(word) > 6]
    return (len(rare_words) / len(content_words) * 100) if content_words else 0

sample_df['rare_word_frequency'] = sample_df.progress_apply(rare_word_frequency, axis=1)
sample_df['modal_verb_patterns'] = sample_df.progress_apply(modal_verb_patterns, axis=1)
sample_df['subordinate_clause_ratio'] = sample_df.progress_apply(subordinate_clause_ratio, axis=1)
sample_df['transition_word_overuse'] = sample_df[text_column].progress_apply(transition_word_overuse)
sample_df['word_length_variance'] = sample_df[text_column].progress_apply(word_length_variance)
sample_df['mean_syllables_per_word'] = sample_df[text_column].progress_apply(average_syllables_per_word)
sample_df['hedging_language_ratio'] = sample_df[text_column].progress_apply(hedging_language_frequency)
sample_df['sentence_length_variance'], sample_df['mean_sentence_length'] = zip(*sample_df[text_column].progress_apply(sentence_length_variance))
sample_df['grammar_issues_per_100_words'] = sample_df.progress_apply(count_grammar_issues_spacy, axis=1)
sample_df['function_word_ratio'] = sample_df[text_column].apply(function_word_frequency)

punct_stats = sample_df[text_column].apply(punctuation_density_diversity)
sample_df['punct_density'] = punct_stats.apply(lambda x: x[0])
sample_df['punct_diversity'] = punct_stats.apply(lambda x: x[1])
sample_df = apply_word_length_stats(sample_df, text_column)

sample_df.to_parquet(f"data/sample_{per_target}_per_target_processed.parquet", index=False, engine='pyarrow')
