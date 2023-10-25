import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk

nltk.download("punkt")

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)  # Tokenize the sentence into words
    n = 0.7
    num_words_to_replace = int(0.7 * len(words))  # Calculate the number of words to replace (n% of total words)

    for _ in range(num_words_to_replace):
        word_idx = random.randint(0, len(words) - 1)  # Randomly select a word index
        word_to_replace = words[word_idx]

        # Get synonyms for the selected word using WordNet
        synonyms = []
        for syn in wordnet.synsets(word_to_replace):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

        if synonyms:
            # Randomly select a synonym to replace the word
            synonym = random.choice(synonyms)
            words[word_idx] = synonym

    # Detokenize the modified words to get the transformed sentence
    transformed_text = TreebankWordDetokenizer().detokenize(words)

    # Update the example with the transformed text
    example["text"] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example
