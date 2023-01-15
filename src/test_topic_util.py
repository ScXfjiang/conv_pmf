from common.topic_util import SparseTokenCountMat
from common.topic_util import NPMI
from common.dictionary import GloveDict6B
import pickle as pkl
import scipy.sparse


if __name__ == "__main__":
    token_cnt_mat = scipy.sparse.load_npz("/ichec/work/ucd01/xfjiang/dataset/amazon/amazon_grocery_and_gourmet_foods1/token_cnt_mat.npz")
    npmi = NPMI(token_cnt_mat)
    with open("/ichec/home/users/xfjiang/workspace/repos/conv_pmf/scripts/ichec/Jan-12-2023-14-51-27-a48b1ce4-0e62-4c11-9feb-d0a8327cfe39/factor2sorted_words.pkl", "rb") as f:
        factor2sorted_topics = pkl.load(f)
    x = npmi.compute_npmi(factor2sorted_topics)
    print(x)
