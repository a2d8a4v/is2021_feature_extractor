from g2p_seq2seq_pytorch.g2p import G2PPytorch
from utils import dict_miss_words


class G2PModel(object):
    def __init__(self):
        # G2P related
        self.g2p_model = G2PPytorch()
        self.g2p_model.load_model()

    def g2p(self, text_predicted, word2phn_dict):
        missed_words = dict_miss_words(text_predicted, word2phn_dict)
        for w in missed_words:
            _ph = self.g2p_model.decode_word(w)
            word2phn_dict[w] = [p.lower() for p in _ph.split()]
        return word2phn_dict