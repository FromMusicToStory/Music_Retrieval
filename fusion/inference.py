import numpy as np
import torch
import torchaudio
from fusion.metric_learning import MLEmbedModel
from fusion.metric_embedding_dataloader import create_data_loader
from text.dataloader import *
from sklearn.metrics.pairwise import *


def example_model_setting(model_path):
    model = torch.load(model_path)
    model.eval()

    return model


def encode(model, inputs):
    embeddings = model.text_encoder(inputs['input_ids'], inputs['mask'])
    return embeddings


def main():
    model_path = "../train/result/ml_with_embed_train_data_epoch9_loss0.2000.pt"
    model = example_model_setting(model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    AUDIO_DIR = '../dataset/mtg-jamendo-dataset/'
    TEXT_DIR = '../dataset/Story_dataset/'
    BATCH_SIZE = 10
    MAX_LEN = 512
    AUDIO_MAX = 500

    data_loader = create_data_loader(AUDIO_DIR, TEXT_DIR, 'test', MAX_LEN, AUDIO_MAX, BATCH_SIZE)
    example = next(iter(data_loader))

    with torch.no_grad():
        audio_embed, text_positive_embed, text_negative_embed, losses = model(example)

    # sentence_embedding = torch.mean(text_positive_embed, dim=0)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 5
    embeddings = []

    dataset = StoryTextDataset(TEXT_DIR, 'test', MAX_LEN)

    for i in range(5):
        with torch.no_grad():
            text = dataset.data[i]['recite_src']['text']
            tokenized =  dataset.tokenizer.encode_plus(text,
                                                       add_special_tokens=True,
                                                       max_length=MAX_LEN,
                                                       padding='max_length',
                                                       truncation=True,
                                                       return_tensors='pt')

            embedding = model.text_encoder(tokenized['input_ids'].to(device),
                                           tokenized['attention_mask'].to(device))

            embedding = embedding.cpu()
            embeddings.append(embedding[0])

    embeddings = np.array(embeddings).squeeze(axis=1)
    print(embeddings.shape)

    cos_scores = cosine_similarity(text_positive_embed[0], embeddings)
    cos_scores = cos_scores.cpu().detach().numpy()

    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", example['anchor'])
    print("\nTop 5 most similar sentences in corpus by emotion mapping:")

    for idx in top_results[0:top_k]:
        print(dataset.decode(dataset.data[idx]['recite_src']['text']), "(Score: %.4f)" % (cos_scores[idx]))

if __name__ == '__main__':
    main()