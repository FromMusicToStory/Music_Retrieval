import gensim.downloader
import gensim
import torch.nn as nn
import torch
import csv

class W2V(nn.Module):
    def __init__(self,download = False ):

        self.model = gensim.downloader.load('word2vec-google-news-300')


    def get_vector(self, word):
        if word in self.model:
            return torch.from_numpy(self.model[word])
        else:
            return None

    def get_cos_sim(self, w1, w2):
        sim = nn.CosineSimilarity(dim=0)
        return sim(w1,w2)

if __name__=="__main__":
    # w2v_model = W2V(download=True)
    w2v_model = W2V()
    story_labels = ["happy","flustered","neutral","angry","anxious","hurt","sad"]


    music_labels = ["action","adventure","advertising","ambient","background","ballad",\
                    "calm","children","christmas","commercial","cool","corporate","dark","deep",\
                    "documentary","drama","dramatic","dream","emotional","energetic","epic","fast",\
                    "film","fun","funny","game","groovy","happy","heavy","holiday","hopeful","horror",\
                    "inspiring","love","meditative","melancholic","mellow","melodic","motivational","movie",\
                    "nature","party","positive","powerful","relaxing","retro","romantic","sad","sexy","slow",\
                    "soft","soundscape","space","sport","summer","trailer","travel","upbeat","uplifting"]

    sims = dict()
    for idx, story in enumerate(story_labels):
        tmp = dict()
        story_emb = w2v_model.get_vector(story)
        for music in music_labels:
            music_emb= w2v_model.get_vector(music)
            if music_emb is None:
                cos_sim=-1
                print(music)
            else:
                cos_sim = w2v_model.get_cos_sim(story_emb, music_emb)
            tmp[music]=cos_sim
        sorted_sims = sorted(tmp.items(), key = lambda item : item[1], reverse=True)
        # (highest, lowest)
        sims[story]=(sorted_sims[0][0], sorted_sims[-1][0])
        print(f'{idx}th label done')

    with open("./cloest_word.csv", 'w') as file:
        writer = csv.DictWriter(file,fieldnames=['story_label','music_label'])
        writer.writeheader()
        for key, value in sims.items():
            writer.writerow({'story_label': key, 'music_label':sims.items()})