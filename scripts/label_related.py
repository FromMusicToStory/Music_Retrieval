from dataset.w2v import W2V
import csv

import json
from collections import defaultdict


## Music Dataset tag label
MTAT = ['singer', 'harpsichord', 'sitar', 'heavy', 'foreign',
       'no piano', 'classical', 'female', 'jazz', 'guitar', 'quiet', 'solo',
       'folk', 'ambient', 'new age', 'synth', 'drum', 'bass', 'loud', 'string',
       'opera', 'fast', 'country', 'violin', 'electro', 'trance', 'chant',
       'strange', 'modern', 'hard', 'harp', 'pop', 'female vocal', 'piano',
       'orchestra', 'eastern', 'slow', 'male', 'vocal', 'no singer', 'india',
       'rock', 'dance', 'cello', 'techno', 'flute', 'beat', 'soft', 'choir',
       'baroque']

Jamendo = ["action","adventure","advertising","ambiental","background","ballad",
           "calm","children","christmas","commercial","cool","corporate","dark","deep",
           "documentary","drama","dramatic","dream","emotional","energetic","epic","fast",
           "film","fun","funny","game","groovy","happy","heavy","holiday","hopeful","horror",
           "inspiring","love","meditative","melancholic","mellow","melodic","motivational","movie",
           "nature","party","positive","powerful","relaxing","retro","romantic","sad","sexy","slow",
           "soft","soundscape","space","sport","summer","trailer","travel","upbeat","uplifting"]


def make_label():

    label = defaultdict(list)

    ## Positive
    # 1. happy
    happy = defaultdict(list)
    happy['positive'] = ["dance", "techno", "pop", "action", "happy", "fun", "funny", "game", "groovy",
                         "holiday", "energetic" "positive", "children", "upbeat", "uplifting", "christmas",
                         "party", "travel", "summer"]

    # 2. neutral
    neutral = defaultdict(list)
    neutral["positive"] = ["calm", "ambiental", "ambient", "ballad", "background", "new age", "classical",
                           "relaxing", "soft", "nature", "meditative", "soundscape", "quiet"]

    # 3. flustered (flustered + anxious)
    flustered = defaultdict(list)
    flustered['positive'] = ["strange", "melancholic", "drama", "hopeful", "inspiring", "melodic"]

    # 4. angry
    angry = defaultdict(list)
    angry["positive"] = ["heavy", "dark", "loud", "rock", "powerful", "beat", ]

    # 5. sad (hurt + sad)
    sad = defaultdict(list)
    sad["positive"] = ["string", "sad", "deep", "ballad", "emotional", "slow"]


    ## Negative
    # 1. happy - neg: flustered, angry, sad
    happy["negative"] = [i for i in (flustered['positive'] + angry['positive'] + sad['positive']) if i not in happy['positive']]

    # 2. neutral - neg: 나머지 다
    neutral["negative"] = [i for i in (happy['positive'] + flustered['positive'] + angry['positive'] + sad['positive']) if i not in neutral['positive']]

    # 3. flustered - neg: happy, neutral
    flustered["negative"] = [i for i in (happy['positive'] + neutral['positive']) if i not in flustered['positive']]

    # 4. angry - neg: happy, neutral, sad
    angry["negative"] = [i for i in (happy['positive'] + neutral['positive'] + sad['positive']) if i not in angry['positive']]

    # 5. sad - neg: happy, angry
    sad["negative"] = [i for i in (happy['positive'] + angry['positive']) if i not in sad['positive']]


    ## Final Label
    label['happy'] = happy
    label['neutral'] = neutral
    label['flustered'] = flustered
    label['angry'] = angry
    label['sad'] = sad
    #label['music'] = defaultdict(list, neutral=neutral)

    return label


def w2v_distance():
    w2v_model = W2V()
    story_labels = ["happy", "flustered", "neutral", "angry", "anxious", "hurt", "sad"]
    music_labels = ["action", "adventure", "advertising", "ambient", "background", "ballad", \
                    "calm", "children", "christmas", "commercial", "cool", "corporate", "dark", "deep", \
                    "documentary", "drama", "dramatic", "dream", "emotional", "energetic", "epic", "fast", \
                    "film", "fun", "funny", "game", "groovy", "happy", "heavy", "holiday", "hopeful", "horror", \
                    "inspiring", "love", "meditative", "melancholic", "mellow", "melodic", "motivational", "movie", \
                    "nature", "party", "positive", "powerful", "relaxing", "retro", "romantic", "sad", "sexy", "slow", \
                    "soft", "soundscape", "space", "sport", "summer", "trailer", "travel", "upbeat", "uplifting"]
    sims = dict()
    for idx, story in enumerate(story_labels):
        tmp = dict()
        story_emb = w2v_model.get_vector(story)
        for music in music_labels:
            music_emb = w2v_model.get_vector(music)
            if music_emb is None:
                cos_sim = -1
                print(music)
            else:
                cos_sim = w2v_model.get_cos_sim(story_emb, music_emb)
            tmp[music] = cos_sim
        sorted_sims = sorted(tmp.items(), key=lambda item: item[1], reverse=True)
        # (highest, lowest)
        sims[story] = (sorted_sims[0][0], sorted_sims[-1][0])
        print(f'{idx}th label done')
    with open("./cloest_word.csv", 'w') as file:
        writer = csv.DictWriter(file, fieldnames=['story_label', 'music_label'])
        writer.writeheader()
        for key, value in sims.items():
            writer.writerow({'story_label': key, 'music_label': sims.items()})