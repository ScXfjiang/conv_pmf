import json
import numpy as np
import nltk
from scipy.sparse import lil_matrix

from common.dictionary import DictionaryIf


def gen_sparse_token_cnt_mat(ref_corpus, dictionary):
    """
    Create sparse token count matrix.
    rows - documents
    cols - tokens

    Args:
        ref_corpus (str): reference corpus
        dictionary (dictionary.DictionaryIf):
    """
    assert isinstance(dictionary, DictionaryIf)

    with open(ref_corpus, "rb") as f:
        num_doc = len(f.readlines())
        voc_size = dictionary.vocab_size()
        token_cnt_mat = lil_matrix((num_doc, voc_size), dtype=np.int32)
        f.seek(0)
        for doc_idx, line in enumerate(f):
            if doc_idx % 1000 == 0:
                print("doc_idx: {}/{}".format(doc_idx, num_doc))
            tokens = nltk.word_tokenize(str(json.loads(line)["tokenized_text"]))
            for token in tokens:
                token_idx = dictionary.word2idx(token)
                token_cnt_mat[doc_idx, token_idx] += 1

        return token_cnt_mat.tocsr()


class NPMIUtil:
    def __init__(self, token_cnt_mat):
        # sparse matrix: [num_doc, voc_size]
        self.token_bool_mat = token_cnt_mat.tocsc().astype('bool')

    def compute_npmi(self, factor2sorted_topics):
        """
        Args:
            factor2sorted_topics (dict): factor_id -> list of topics
        """
        # npmi for each factor [n_factor,]
        factor2npmi = {}
        npmi_cache = {}
        for factor, sorted_topics in factor2sorted_topics.items():
            # no extracted topics
            if len(sorted_topics) == 0:
                factor2npmi[factor] = 0.0
            else:
                # npmi for each (topic_i, topic_j) pair
                pair_npmis = []
                for i, topic_i in enumerate(sorted_topics):
                    for topic_j in sorted_topics[i + 1 :]:
                        ij = frozenset([topic_i, topic_j])
                        if ij in npmi_cache:
                            npmi = npmi_cache[ij]
                        else:
                            col_i = self.token_bool_mat[:, topic_i]
                            col_j = self.token_bool_mat[:, topic_j]
                            # count of topic_i
                            c_i = col_i.sum()
                            # count of topic_j
                            c_j = col_j.sum()
                            # count of (topic_i, topic_j)
                            c_ij = col_i.multiply(col_j).sum()
                            if c_ij == 0:
                                npmi = 0.0
                            else:
                                num_doc = self.token_bool_mat.shape[0]
                                npmi = (
                                    np.log2(num_doc)
                                    + np.log2(c_ij)
                                    - np.log2(c_i)
                                    - np.log2(c_j)
                                ) / (np.log2(num_doc) - np.log2(c_ij))
                            npmi_cache[ij] = npmi
                        pair_npmis.append(npmi)
                factor2npmi[factor] = np.mean(pair_npmis)
        return factor2npmi
