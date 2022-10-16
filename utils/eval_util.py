import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import average_precision_score, accuracy_score, f1_score


def topic_diversity(topic_matrix, top_k=25):
    """ Topic Diversity (TD) measures how diverse the discovered topics are.

    We define topic diversity to be the percentage of unique words in the top 25 words (Dieng et al., 2020)
    of the selected topics. TD close to 0 indicates redundant topics, TD close to 1 indicates more varied topics.

    Args:
        topic_matrix:
        top_k:
    """
    num_topics = topic_matrix.shape[0]
    top_words_idx = np.zeros((num_topics, top_k))
    for k in range(num_topics):
        idx = np.argsort(topic_matrix[k, :])[::-1][:top_k]
        top_words_idx[k, :] = idx
    num_unique = len(np.unique(top_words_idx))
    num_total = num_topics * top_k
    td = num_unique / num_total
    # print('Topic diversity is: {}'.format(td))
    return td


def compute_npmi(corpus, word_i, word_j):
    """ Pointwise Mutual Information (PMI) measures the association of a pair of outcomes x and y.

    PMI is defined as log[p(x, y)/p(x)p(y)], which can be further normalized between [-1, +1], resulting in
    -1 (in the limit) for never occurring together, 0 for independence, and +1 for complete co-occurrence.
    The Normalized PMI is computed by PMI(x, y) / [-log(x, y)].
    """
    num_docs = len(corpus)
    num_docs_appear_wi = 0
    num_docs_appear_wj = 0
    num_docs_appear_both = 0
    for n in range(num_docs):
        doc = corpus[n]
        # doc = corpus[n].squeeze(0)
        # doc = [doc.squeeze()] if len(doc) == 1 else doc.squeeze()

        # if word_i in doc:
        if doc[word_i] > 0:
            num_docs_appear_wi += 1
        # if word_j in doc:
        if doc[word_j] > 0:
            num_docs_appear_wj += 1
        # if [word_i, word_j] in doc:
        if doc[word_i] > 0 and doc[word_j] > 0:
            num_docs_appear_both += 1

    if num_docs_appear_both == 0:
        return -1
    else:
        pmi = np.log(num_docs) + np.log(num_docs_appear_both) - \
              np.log(num_docs_appear_wi) - np.log(num_docs_appear_wj)
        return pmi / (np.log(num_docs) - np.log(num_docs_appear_both))


def topic_coherence(corpus, vocab, topic_matrix, top_k=10):
    """ Topic Coherence measures the semantic coherence of top words in the discovered topics.

    We apply the widely-used Normalized Pointwise Mutual Information (NPMI) (Aletras & Stevenson, 2013; Lau et al., 2014)
    computed over the top 10 words of each topic, by the Palmetto package (Röder et al., 2015).

    Args:
        corpus:
        vocab:
        topic_matrix:
        top_k:
    """
    num_docs = len(corpus)
    # print('Number of documents: ', num_docs)

    tc_list = []
    num_topics = topic_matrix.shape[0]
    print('Number of topics: ', num_topics)
    for k in range(num_topics):
        # print('Topic Index: {}/{}'.format(k, num_topics))
        top_words_idx = np.argsort(topic_matrix[k, :])[::-1][:top_k]
        # top_words = [vocab[idx] for idx in list(top_words_idx)]

        pairs_count = 0
        tc_k = 0
        # for i, word in enumerate(top_words):
        for i, word in enumerate(top_words_idx):
            for j in range(i + 1, top_k):
                # tc_list.append(compute_npmi(corpus, word, top_words[j]))
                # tc_list.append(compute_npmi(corpus, word, top_words_idx[j]))
                tc_k += compute_npmi(corpus, word, top_words_idx[j])
                pairs_count += 1
        tc_list.append(tc_k / pairs_count)

    # tc = sum(tc_list) / (num_topics * pairs_count)
    # print('Topic coherence is: {}'.format(tc))
    tc_list.sort(reverse=True)
    half_num = int(num_topics/2)
    return sum(tc_list[: half_num]) / half_num


def text_classification(train_data, train_labels, test_data, test_labels, algorithm='LR'):
    if algorithm == 'LR':
        clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
    elif algorithm == 'SVM':
        clf = SVC(random_state=0)
    else:
        raise NotImplementedError

    clf.fit(train_data, train_labels)
    test_acc = accuracy_score(clf.predict(test_data), test_labels)
    # print('Accuracy on the test set: {}'.format(test_acc))
    return test_acc


def text_clustering(data, labels_true, num_clusters=30):
    # standardization
    mu = np.mean(data, axis=-1, keepdims=True)
    sigma = np.std(data, axis=-1, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1)
    data = (data - mu) / sigma

    # clustering based on Euclidean distance
    estimator = KMeans(n_clusters=num_clusters, random_state=0)
    estimator.fit(data)
    labels_pred = estimator.labels_

    purity_score = purity(labels_true, labels_pred)
    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
    # print('Clustering purity: {}'.format(purity_score))
    # print('Clustering nmi: {}'.format(nmi_score))
    return purity_score, nmi_score


def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    counts = []
    for c in clusters:
        indices = np.where(labels_pred == c)[0]
        max_votes = np.bincount(labels_true[indices]).max()
        counts.append(max_votes)
    return sum(counts) / labels_true.shape[0]

