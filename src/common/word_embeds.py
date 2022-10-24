import numpy as np


def get_embeds_type(type):
    if type == "glove.6B.50d":
        EmbedsT = GloveEmbeds50D
    elif type == "glove.6B.100d":
        EmbedsT = GloveEmbeds100D
    elif type == "glove.6B.200d":
        EmbedsT = GloveEmbeds200D
    elif type == "glove.6B.300d":
        EmbedsT = GloveEmbeds300D
    else:
        raise NotImplementedError

    return EmbedsT


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


class GloveEmbeds50D(GloveEmbeds):
    def __init__(self, path):
        super().__init__(path)


class GloveEmbeds100D(GloveEmbeds):
    def __init__(self, path):
        super().__init__(path)


class GloveEmbeds200D(GloveEmbeds):
    def __init__(self, path):
        super().__init__(path)


class GloveEmbeds300D(GloveEmbeds):
    def __init__(self, path):
        super().__init__(path)
