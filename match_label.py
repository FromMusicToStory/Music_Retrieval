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


if __name__ == "__main__":
    label = make_label()
    print(label.keys())
