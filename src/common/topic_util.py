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
                js = json.loads(line)
                tokenizer = torchtext.data.get_tokenizer("basic_english")
                tokens = tokenizer(str(js["reviewText"]))
                for token in tokens:
                    token_idx = dictionary.word2idx(token)
                    self.token_cnt_mat[doc_idx, token_idx] += 1
            self.token_cnt_mat = self.token_cnt_mat.tocsr()

    def save(self, save_path):
        scipy.sparse.save_npz(save_path, self.token_cnt_mat)


class NPMI:
    pass
