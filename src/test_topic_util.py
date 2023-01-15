from common.topic_util import SparseTokenCountMat
from common.topic_util import NPMI
from common.dictionary import GloveDict6B
import pickle as pkl
import scipy.sparse


if __name__ == "__main__":
    token_cnt_mat = scipy.sparse.load_npz("/ichec/work/ucd01/xfjiang/dataset/amazon/amazon_grocery_and_gourmet_foods1/token_cnt_mat.npz")
    npmi = NPMI(token_cnt_mat)
    with open("/ichec/home/users/xfjiang/workspace/repos/conv_pmf/scripts/ichec/Jan-15-2023-14-51-30-b95f7406-fe49-469d-8fa5-9c4895b575e0/factor2sorted_tokens.pkl", "rb") as f:
        factor2sorted_topics = pkl.load(f)
    x = npmi.compute_npmi(factor2sorted_topics)
    print(x)
