#!/usr/bin/env python3
import os
import pickle
import gensim
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer, SnowballStemmer

'''
Implement the Latent Dirichlet Allocation algorithm as it was described in the lectures.
The attached data in abcnews-date-text.csv consist of 1186018 news headlines.
Consider the headlines as individual short documents. You can use only a subset of them (first 10.000 lines) to make the computations faster.
Use the preprocessing (lemmatization, stemming, filtering dictionary) as proposed at the beginning of the script lda.py.
fill the missing parts of the script lda.py

marecek@ufal.mff.cuni.cz.
'''


# pic_prior = np.full(shape=topic_


def create_weights(d_id, w_id, doc_topic_cnt, doc_top_prior, top_wrd_cnt, top_wrd_prior):
    # document topic like
    d_t = doc_topic_cnt[d_id, :] + doc_top_prior  # cnt of topics 1 .. K in document d
    norm = np.sum(doc_topic_cnt[d_id, :] + doc_top_prior)  # norm: sum over all topics in doc d
    d_t = d_t / norm

    # topic word like [ does not sum to 1 ]
    t_w = top_wrd_cnt[:, w_id] + top_wrd_prior[w_id]  # cnt of w_dn in topics 1 ... K
    norm = np.sum(top_wrd_cnt, axis=1) + np.sum(top_wrd_prior)  # norm: sum all words to topics 1 ... K
    t_w = t_w / norm

    result = np.multiply(d_t, t_w)

    # normalize for sampling
    result = result / sum(result)
    return result


# ##################
# Preprocess ...
# ##################

def pickle_data(obj, path):
    print("Writing pickle: {}".format(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_and_preprocess(filename='abcnews-date-text.csv', cnt=10000):
    # Load if we already processed in order to improve speed ...
    doc_path = "docs_{}.pkl".format(cnt)
    dic_path = "dictionary_{}.pkl".format(cnt)

    if os.path.isfile(doc_path) and os.path.isfile(dic_path):
        print("Loading existing pickle")
        with open(doc_path, 'rb') as handle:
            _doc = pickle.load(handle)

        with open(dic_path, 'rb') as handle:
            _dict = pickle.load(handle)
        return _doc, _dict

    print("Preprocess... ", end="")
    # Load documents
    # You can use only a subset, for example first 10000 documents.
    data = pd.read_csv(filename, error_bad_lines=False, nrows=cnt)
    data_text = data[['headline_text']]
    data_text['index'] = data_text.index
    documents = data_text

    print(len(documents), " documents loaded.")

    # Preprocess documents - lemmatization and stemming
    def lemmatize_stemming(text):
        stemmer = SnowballStemmer("english")
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    print("Preprocess: Lemmatize...")
    processed_docs = documents['headline_text'].map(preprocess)

    # Construct dictionary
    print("Preprocess: Gensim...")
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5)  # in (0.15 , 0.5) documents ; max 100K

    # Filter words in documents == remove ...
    print("Preprocess: Filter...")
    docs = list()
    for doc in processed_docs:
        docs.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))

    pickle_data(docs, doc_path)
    pickle_data(dictionary, dic_path)

    return docs, dictionary


def print_results(doc_topic_cnt, topic_word_cnt, dic, print_cnt_top_words, print_cnt_docs):
    topic_cnt = topic_word_cnt.shape[0]

    def normalize_rows(matrix):
        row_sums = matrix.sum(axis=1)
        # there can be empty document (with all words filtered out) thus document with no topics
        # to prevent division by zero we +1 sums of rows equal to 0 (docs with no topics)
        row_sums[row_sums == 0] = 1
        res = matrix / row_sums[:, np.newaxis]
        return res

    # (3) For each topic, print 10 most frequent words and their frequencies.
    if print_cnt_top_words is not None:
       topic_word_freq = normalize_rows(topic_word_cnt)
       for k in range(topic_cnt):
           print("### t_{} ###".format(k))
           word_ids = np.argsort(-1 * topic_word_freq[k])  # sort from highest ...
           for i in range(min(print_cnt_top_words, len(word_ids))):
               w_id = word_ids[i]
               if topic_word_freq[k][w_id] > 0:
                   print("{:.4f} {}".format(topic_word_freq[k][w_id], dic[w_id]))
           print("")

    # (4) For the first 10 documents, show the distribution across their topics.
    if print_cnt_docs is not None:
        doc_topic_freq = normalize_rows(doc_topic_cnt)
        for d in range(min(print_cnt_docs, len(doc_topic_freq))):
            print("doc_{}: ".format(d), end="")
            wrds = []
            for wid_topid in docs[d]:
                wrds.append((dic[wid_topid[0]], wid_topid[1]))
            print("\t", end="")
            print(wrds)

            topic_ids = np.argsort(-1 * doc_topic_freq[d])  # sort from highest ...
            for k in range(topic_cnt):
                topic_id = topic_ids[k]
                if doc_topic_freq[d][topic_id] > 0:
                    print("\t{:.2f}, t_{}".format(doc_topic_freq[d][topic_id], topic_id))
            print("")


def topic_wrd_df(topic_wrd_count, top_words=10):
    #  create some nice dataframe ...
    topic_cnt = topic_wrd_count.shape[0]
    topics = []
    for i in range(topic_cnt):
        df = pd.DataFrame(data=topic_wrd_count.transpose()[:, i], index=[dictionary[i] for i in range(wrd_cnt)],
                          columns=["t_{}".format(i)])
        df = df[(df.T != 0).any()]  # drop zero
        df = df / df.sum(axis=0)
        df = df.sort_values(by=['t_{}'.format(i)], ascending=False)
        df = df.head(top_words)  # only top 10
        df = df.T
        topics.append(df)
        # and print it ...
    df = pd.concat(topics).T
    return df


def gibbs_sampling(docs, iter_cnt, topic_cnt, wrd_cnt, doc_cnt, alpha, gamma):
    # INITIALIZE
    doc_top_prior = np.full(shape=topic_cnt, fill_value=alpha,
                            dtype=float)  # np.random.dirichlet(alpha, size=topic_cnt)   # 1*topics
    top_wrd_prior = np.full(shape=wrd_cnt, fill_value=gamma,
                            dtype=float)  # np.random.dirichlet(gamma, size=wrd_cnt)  # 1*words
    doc_top_cnt = np.zeros(shape=(doc_cnt, topic_cnt), dtype=int)
    top_wrd_cnt = np.zeros(shape=(topic_cnt, wrd_cnt), dtype=int)

    # Randomly initialize topics for all words in all documents.
    uniform_prob = np.full(shape=topic_cnt, fill_value=1 / topic_cnt, dtype=float)

    for doc_id, doc in enumerate(docs):
        for i, word_id in enumerate(doc):
            # sample random topic
            topic_id = np.argmax(np.random.multinomial(n=1, pvals=uniform_prob))
            # and edit document from 'word_id' -> [word_id, topic_ic]
            doc[i] = [word_id, topic_id]
            # finally, increase counts
            doc_top_cnt[doc_id, topic_id] += 1
            top_wrd_cnt[topic_id, word_id] += 1

    # ITERATIONS
    print("GIBBS:")
    print("iter: {} ...\r".format(0), end="")
    for it in range(iter_cnt):
        changed_topics = 0
        total_wrds = 0

        # for each word in each doc ...
        for dID, doc in enumerate(docs):
            for wrdID_topic in doc:
                total_wrds += 1
                wrd_id = wrdID_topic[0]
                old_topic_id = wrdID_topic[1]

                # decrement old
                assert doc_top_cnt[dID, old_topic_id] > 0, "doc_topic_cnt[doc_id, old_topic_id] == 0"
                assert top_wrd_cnt[old_topic_id, wrd_id] > 0, "top_wrd_cnt[old_topic_id, wrd_id] == 0"
                doc_top_cnt[dID, old_topic_id] -= 1
                top_wrd_cnt[old_topic_id, wrd_id] -= 1

                # Create weights and sample new topic
                p = create_weights(dID, wrd_id, doc_top_cnt, doc_top_prior, top_wrd_cnt, top_wrd_prior)
                new_topic_id = np.argmax(np.random.multinomial(n=1, pvals=p))
                # print(p)
                # print(new_topic_id)

                # increment new
                doc_top_cnt[dID, new_topic_id] += 1
                top_wrd_cnt[new_topic_id, wrd_id] += 1
                wrdID_topic[1] = new_topic_id
                if old_topic_id != new_topic_id:
                    changed_topics += 1
        print("iter: {} changed_words: {} [{:.2f}]".format(it, changed_topics, changed_topics / total_wrds))
    print("\n")
    return doc_top_cnt, top_wrd_cnt


if __name__ == "__main__":
    np.random.seed(2018)
    nltk.download('wordnet')

    # LOAD DATA
    cnt = 10000  # 10K
    docs, dictionary = load_and_preprocess(filename='abcnews-date-text.csv', cnt=cnt)
    doc_cnt = len(docs)
    wrd_cnt = len(dictionary)

    # SET HYPER PARAMS
    topic_cnt = 10
    iterations = 10
    alpha = 0.1  # doc topic prior
    gamma = 0.00001  # topic word prior

    # GIBBS SAMPLING
    doc_top_count, topic_wrd_count = gibbs_sampling(docs=docs, iter_cnt=iterations, topic_cnt=topic_cnt,
                                                    wrd_cnt=wrd_cnt, doc_cnt=doc_cnt, alpha=alpha, gamma=gamma)

    print_results(doc_top_count, topic_wrd_count, dictionary, print_cnt_top_words=10, print_cnt_docs=10)