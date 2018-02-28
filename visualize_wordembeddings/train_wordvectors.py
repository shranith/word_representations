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
import argparse
from nltk import word_tokenize


def load_corpus(corpus):
    book_filenames = sorted(glob.glob(corpus+"/*.txt"))
    print(book_filenames)
    corpus_raw = u""
    for book_filename in book_filenames:
        print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        print("Corpus is now {0} characters long".format(len(corpus_raw)))
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus_raw)
    #sentence where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))
    token_count = sum([len(sentence) for sentence in sentences])
    print("The book corpus contains {0:,} tokens".format(token_count))
    return sentences
# convert into a list of words
# remove unnnecessary,, split into words, no hyphens
# list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

def train_embeddings(sentences):
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

def loadModel_and_visualize(model_folder):
    thrones2vec = w2v.Word2Vec.load(os.path.join(model_folder, "thrones2vec.w2v"))
    # tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    # all_word_vectors_matrix = thrones2vec.wv.syn0
    # all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
    # points = pd.DataFrame(
    #     [
    #         (word, coords[0], coords[1])
    #         for word, coords in [
    #             (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
    #             for word in thrones2vec.wv.vocab
    #         ]
    #     ],
    #     columns=["word", "x", "y"]
    # )
    # print(points.head(10))
    # sns.set_context("poster")
    # points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    # print(points.plot.scatter("x", "y", s=10, figsize=(20, 12)))
    # plt.show()
    return thrones2vec

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

# plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))
# plt.show()
# plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))


# print(thrones2vec.most_similar("Stark"))


def nearest_similarity_cosmul(start1, end1, end2, thrones2vec):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))

def parse_args():
    """ Parse the command line arguments into their proper collections. """
    parser = argparse.ArgumentParser(
                    description="""Train and visualize word embeddings""",
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        help="select train or predict mode. Please provide the following for train: corpus folder, a folder name to save the trained model")
    parser.add_argument("--corpus",
                        type=str,
                        default="data",
                        help="corpus to train word embeddings on, it expects txt files")
    parser.add_argument("--model_folder",
                        type=str,
                        default="model",
                        help="folder to save the train w2c model")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        sentences = load_corpus(args.corpus)
        train_embeddings(sentences,args.model_folder)
    elif args.mode == "predict":
        thrones2vec =  loadModel_and_visualize(args.model_folder)
    else:
        print("Invalid mode, select either train or predict\n")
        sys.exit()
    
    print(thrones2vec.most_similar("Stark"))
    nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun", thrones2vec)
    nearest_similarity_cosmul("Jaime", "sword", "wine", thrones2vec)
    nearest_similarity_cosmul("Arya", "Nymeria", "dragons", thrones2vec)



if __name__ == "__main__":
    main()

