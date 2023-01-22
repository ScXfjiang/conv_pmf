import numpy as np


class EmbedsIf(object):
    def __init__(self, path):
        super().__init__()

    def embed_matrix(self):
        raise NotImplementedError

    def embed_dim(self):
        raise NotImplementedError


class GloveEmbeds(EmbedsIf):
    def __init__(self, path):
        super().__init__(path)
        self.__embed_matrix = []
        with open(path, "r") as f:
            for _, line in enumerate(f):
                embed = line.rstrip().split(" ")[1:]
                embed = list(map(lambda x: float(x), embed))
                self.__embed_matrix.append(embed)
        self.__embed_matrix = np.array(self.__embed_matrix, dtype=np.float32)
        # unknown embedding
        self.__embed_matrix = np.concatenate(
            (self.__embed_matrix, self.__embed_matrix.mean(0, keepdims=True))
        )
        # padding embedding
        self.__embed_matrix = np.concatenate(
            (
                self.__embed_matrix,
                np.zeros((1, self.__embed_matrix.shape[1]), dtype=np.float32),
            )
        )

    def embed_matrix(self):
        return self.__embed_matrix

    def embed_dim(self):
        return self.__embed_matrix.shape[1]

