# Music_Retrieval

- Cross-modal Music-to-Story Retrieval task
- Retrieving Story from Music
- With emotion labels, we tried to exploit embedding spaces for mapping story & music.

<br>

## Model Architecture

<img src = "/img/Model Architecture.png">

### Query (Audio) Encoder
- ① GST style Reference Encoder
<img src = "/img/Reference%20Encoder.PNG">
- ② VAE style Reference Encoder
<img src = "/img/Reference%20Encoder_VAE.PNG">

<br>

## Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-lNsxA9rHXzKLF21S4vWmtvNT1MmLk5G?usp=sharing)

<br>

## Set up Environment
```
pip install -r requirements.txt
```

<br>

## Dataset
#### - Text: NIA의 구연동화 음성합성기 훈련 데이터
- Sentences from poetry, novels, dramas, scenarios, etc.
- Seven emotion labels (happy, neutral, flustered, anxious, angry, sad, hurt)
#### - Audio: MTG-Jamendo dataset
- https://mtg.github.io/mtg-jamendo-dataset
- You need to download audio files for the ```autotagging_moodtheme.tsv``` subset.
- Many moodtheme tags (action,adventure,advertising,ambiental,background, ballad...)
- So, we manually map music labels onto story labels 

<br>

## Training

1. two branch metric learning
```
bash ./run_twobranch_train.sh
```

2. three branch metric learning
```
bash ./run_train.sh
```
