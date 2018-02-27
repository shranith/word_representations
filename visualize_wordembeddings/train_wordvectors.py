from __future__ import absolute_import, division, print_function
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk import word_tokenize

def load_corpus():
    book_filenames = sorted(glob.glob("data/*.txt"))
    print(book_filenames)
    corpus_raw = u""
    for book_filename in book_filenames:
        print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        print("Corpus is now {0} characters long".format(len(corpus_raw)))
        print()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus_raw)
    #sentence where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))
    token_count = sum([len(sentence) for sentence in sentences])
    print("The book corpus contains {0:,} tokens".format(token_count))

# convert into a list of words
# remove unnnecessary,, split into words, no hyphens
# list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words




def train_embeddings():
    # ONCE we have vectors
    # step 3 - build model
    # 3 main tasks that vectors help with
    # DISTANCE, SIMILARITY, RANKING

    # Dimensionality of the resulting word vectors.
    # more dimensions, more computationally expensive to train
    # but also more accurate
    # more dimensions = more generalized
    num_features = 300
    # Minimum word count threshold.
    min_word_count = 3

    # Number of threads to run in parallel.
    #more workers, faster we train
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = 7

    # Downsample setting for frequent words.
    #0 - 1e-5 is good for this
    downsampling = 1e-3

    # Seed for the RNG, to make the results reproducible.
    #random number generator
    #deterministic, good for debugging
    seed = 1

    thrones2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    thrones2vec.build_vocab(sentences)


    print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))

    thrones2vec.train(sentences,total_examples=thrones2vec.corpus_count,epochs=thrones2vec.epochs)

    if not os.path.exists("trained"):
        os.makedirs("trained")

    thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))




def loadModel_and_visualize():
    thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    all_word_vectors_matrix = thrones2vec.wv.syn0
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
                for word in thrones2vec.wv.vocab
            ]
        ],
        columns=["word", "x", "y"]
    )
    print(points.head(10))
    sns.set_context("poster")
    points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    print(points.plot.scatter("x", "y", s=10, figsize=(20, 12)))
    plt.show()


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))
plt.show()
# plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))


print(thrones2vec.most_similar("Stark"))


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2



def main():


    
    print(nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun"))
    print(nearest_similarity_cosmul("Jaime", "sword", "wine"))
    print(nearest_similarity_cosmul("Arya", "Nymeria", "dragons"))



if __name__ == "__main__":
    main()

