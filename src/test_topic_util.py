from common.topic_util import SparseTokenCountMat
from common.topic_util import NPMI
from common.dictionary import GloveDict6B


if __name__ == "__main__":
    dataset_path = "/ichec/work/ucd01/xfjiang/dataset/amazon/amazon_grocery_and_gourmet_foods1/train.json"
    dictionary = GloveDict6B(
        "/ichec/work/ucd01/xfjiang/dataset/glove.6B/glove.6B.50d.txt"
    )
    token_cnt_mat = SparseTokenCountMat(dataset_path, dictionary)
    token_cnt_mat.save("token_cnt_mat.npz")