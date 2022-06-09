import gensim.downloader
import gensim
import torch.nn as nn

class W2V():
    def __init__(self,download = False ):

        self.model = gensim.downloader.load('word2vec-google-news-300')

    def get_vector(self, word):
        if word in self.model:
            return self.model[word]
        else:
            return None

    def get_cos_sim(self, w1, w2):
        return nn.CosineSimilarity(w1, w2)

if __name__=="__main__":
    # w2v_model = W2V(download=True)
    w2v_model = W2V()
    print(w2v_model.get_vector("word"))
