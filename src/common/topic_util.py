import json
import numpy as np
import torchtext
import scipy

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
            token_cnt_mat = np.zeros((num_doc, voc_size), dtype=np.int32)
            f.seek(0)
            for doc_idx, line in enumerate(f):
                js = json.loads(line)
                tokenizer = torchtext.data.get_tokenizer("basic_english")
                tokens = tokenizer(str(js["reviewText"]))
                for token in tokens:
                    token_idx = dictionary.word2idx(token)
                    token_cnt_mat[doc_idx][token_idx] += 1

        self.sparse_token_cnt_mat = scipy.sparse.csr_matrix(token_cnt_mat)

    def save(self, save_path):
        scipy.sparse.save_npz(save_path, self.sparse_token_cnt_mat)


class NPMI:
    pass
