def get_dictionary_type(type):
    if type == "glove.6B.50d":
        DictionaryT = GloveDict6B50D
    elif type == "glove.6B.100d":
        DictionaryT = GloveDict6B100D
    elif type == "glove.6B.200d":
        DictionaryT = GloveDict6B200D
    elif type == "glove.6B.300d":
        DictionaryT = GloveDict6B300D
    else:
        raise NotImplementedError

    return DictionaryT


class DictionaryIf(object):
    def __init__(self, path):
        super(DictionaryIf, self).__init__()

    def word2idx(self, word):
        raise NotImplementedError

    def idx2word(self, idx):
        raise NotImplementedError

    def vocab_size(self):
        raise NotImplementedError

    def unknown_idx(self):
        raise NotImplementedError

    def padding_idx(self):
        raise NotImplementedError


class GloveDict6B(DictionaryIf):
    def __init__(self, path):
        super(GloveDict6B, self).__init__(path)
        self.__word2idx = {}
        self.__idx2word = {}
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                word = str(line.rstrip().split(" ")[0])
                self.__word2idx[word] = idx
                self.__idx2word[idx] = word
        # -2: unknown word, -1: padding word
        self.__vocab_size = len(self.__word2idx) + 2

    def word2idx(self, word):
        if word in self.__word2idx:
            return self.__word2idx[word]
        else:
            return self.unknown_idx()

    def idx2word(self, idx):
        if idx > self.padding_idx():
            assert False, "out of index"
        elif idx == self.padding_idx():
            return "<PAD>"
        elif idx == self.unknown_idx():
            return "<UNK>"
        else:
            return self.__idx2word[idx]

    def vocab_size(self):
        return self.__vocab_size

    def unknown_idx(self):
        return self.vocab_size() - 2

    def padding_idx(self):
        return self.vocab_size() - 1


class GloveDict6B50D(GloveDict6B):
    def __init__(self, path):
        super().__init__(path)


class GloveDict6B100D(GloveDict6B):
    def __init__(self, path):
        super().__init__(path)


class GloveDict6B200D(GloveDict6B):
    def __init__(self, path):
        super().__init__(path)


class GloveDict6B300D(GloveDict6B):
    def __init__(self, path):
        super().__init__(path)
