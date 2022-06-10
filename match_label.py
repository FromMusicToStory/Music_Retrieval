import json
from collections import defaultdict


MTAT_tag = ['singer', 'harpsichord', 'sitar', 'heavy', 'foreign',
       'no piano', 'classical', 'female', 'jazz', 'guitar', 'quiet', 'solo',
       'folk', 'ambient', 'new age', 'synth', 'drum', 'bass', 'loud', 'string',
       'opera', 'fast', 'country', 'violin', 'electro', 'trance', 'chant',
       'strange', 'modern', 'hard', 'harp', 'pop', 'female vocal', 'piano',
       'orchestra', 'eastern', 'slow', 'male', 'vocal', 'no singer', 'india',
       'rock', 'dance', 'cello', 'techno', 'flute', 'beat', 'soft', 'choir',
       'baroque']

Jamendo_mood = ["action","adventure","advertising","ambiental","background","ballad",
                "calm","children","christmas","commercial","cool","corporate","dark","deep",
                "documentary","drama","dramatic","dream","emotional","energetic","epic","fast",
                "film","fun","funny","game","groovy","happy","heavy","holiday","hopeful","horror",
                "inspiring","love","meditative","melancholic","mellow","melodic","motivational","movie",
                "nature","party","positive","powerful","relaxing","retro","romantic","sad","sexy","slow",
                "soft","soundscape","space","sport","summer","trailer","travel","upbeat","uplifting"]


label = defaultdict(list)

label['MTAT']
label["Jamendo"]


## 1. happy
# neg: 나머지 다
happy = defaultdict(list)
happy['positive'] = ["dance", "techno", "pop", "action", "happy", "fun", "funny", "game",
                     "groovy", "holiday", "energetic" "positive", "children", "upbeat", "uplifting", "christmas", "party",
                     "travel", "summer"]
#happy["negative"] = ["dark", "horror", "sad", "ballad", "melancholic"]


## 2. flustered + anxious = flustered
# neg: happy, neutral
flustered = defaultdict(list)
flustered['positive'] = ["strange", "adventure", "fast"]
#flustered["negative"] = ["happy", "cool", "fun", ]

anxious = defaultdict(list)
anxious["positive"] =  ["hopeful"]
#anxious["negative"] =  []

## 3. neutral
# neg: 나머지 다
neutral = defaultdict(list)
neutral["positive"] =  ["calm", "ambiental", "ambient", "ballad", "background", "new age", "classical",
                        "relaxing", "soft", "nature", "meditative", "soundscape"]
#neutral["negative"] =  ["action", "drama", "dramatic", "emotional", "energetic", "fast", "game", "sport"]

## 4. angry
# neg: happy, neutral, sad
angry = defaultdict(list)
angry["positive"] =  ["heavy", "dark", "loud", "heavy", "rock", "powerful", "beat"]
#angry["negative"] =  []


## 5. hurt + sad = sad
# neg: happy, angry
hurt = defaultdict(list)
hurt["positive"] =  ["string", "sad", "melancholic"]
#hurt["negative"] =  []

sad = defaultdict(list)
sad["positive"] =  ["deep", "ballad", "melancholic", "emotional", "sad", "slow"]
#sad["negative"] =  []



