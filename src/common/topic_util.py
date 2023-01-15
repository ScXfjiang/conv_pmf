import json
import numpy as np
import torchtext
import scipy
from scipy.sparse import lil_matrix

from common.dictionary import DictionaryIf


class SparseTokenCountMat:
    def __init__(self, dataset_path, dictionary):
        """
        Args:
            dataset_path (str): Amazon dataset
            dictionary (dictionary.DictionaryIf):
        """
        assert isinstance(dictionary, DictionaryIf)

        with open(dataset_path, "rb") as f:
            num_doc = len(f.readlines())
            voc_size = dictionary.vocab_size()
            self.token_cnt_mat = lil_matrix((num_doc, voc_size), dtype=np.int32)
            f.seek(0)
            for doc_idx, line in enumerate(f):
                tokenizer = torchtext.data.get_tokenizer("basic_english")
                tokens = tokenizer(str(json.loads(line)["reviewText"]))
                for token in tokens:
                    token_idx = dictionary.word2idx(token)
                    self.token_cnt_mat[doc_idx, token_idx] += 1
            self.token_cnt_mat = self.token_cnt_mat.tocsr()

    def save(self, save_path):
        scipy.sparse.save_npz(save_path, self.token_cnt_mat)


class NPMI:
    def __init__(self, token_cnt_mat):
        # sparse matrix: [num_doc, voc_size]
        self.token_cnt_mat = token_cnt_mat
        self.npmi_cache = {}

    def compute_npmi(self, factor2sorted_topics, k=10):
        """
        Args:
            factor2sorted_topics (dict): factor_id -> list of topics
            k (int): top_k topics
        """
        # npmi for each factor [num_doc,]
        npmi_means = []
        for _, sorted_topics in factor2sorted_topics.items():
            if len(sorted_topics) > k:
                sorted_topics = sorted_topics[:k]
            npmi_vals = []
            for i, topic_i in enumerate(sorted_topics):
                for topic_j in sorted_topics[i + 1 :]:
                    ij = frozenset([topic_i, topic_j])
                    if ij in self.npmi_cache:
                        npmi = self.npmi_cache[ij]
                    else:
                        col_i = self.token_cnt_mat[:, topic_i]
                        col_j = self.token_cnt_mat[:, topic_j]
                        c_i = col_i.sum()
                        c_j = col_j.sum()
                        c_ij = col_i.multiply(col_j).sum()
                        if c_ij == 0:
                            npmi = 0.0
                        else:
                            num_doc = self.token_cnt_mat.shape[0]
                            npmi = (
                                np.log(num_doc)
                                + np.log(c_ij)
                                - np.log(c_i)
                                - np.log(c_j)
                            ) / (np.log(num_doc) - np.log(c_ij))
                        self.npmi_cache[ij] = npmi
                    npmi_vals.append(npmi)
            npmi_means.append(np.mean(npmi_vals))
        return np.array(npmi_means)

