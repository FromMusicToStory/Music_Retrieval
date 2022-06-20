import numpy as np
import torch
import torchaudio
from fusion.metric_learning import MLEmbedModel
from fusion.metric_embedding_dataloader import JamendoDataset, StoryTextDataset
from text.dataloader import *
from sklearn.metrics.pairwise import *

from multi_model import MetricModel
from multi_dataloader import AudioOnlyDataset
import argparse


def example_model_setting(model, model_path,which_model):
    checkpoint = torch.load(model_path)
    if which_model=='mlembed':
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def encode(model, inputs):
    embeddings = model.text_encoder(inputs['input_ids'], inputs['mask'])
    return embeddings


from multi_dataloader import AudioTextDataset as MultiDataset
from torch.utils.data import DataLoader


def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoint/metric_224_val_loss_1.00000.pth')
    parser.add_argument('--model', type=str, default='metric', help='select model to infer, mlembed or metric')

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    AUDIO_DIR = './dataset/MTG/'
    TEXT_DIR = './dataset/Story_dataset/'
    BATCH_SIZE = 10
    MAX_LEN = 512
    AUDIO_MAX = 500

    if args.model == 'mlembed':
        dataset = JamendoDataset(AUDIO_DIR, TEXT_DIR, 'test', MAX_LEN, AUDIO_MAX, filter_audio_save=True)
        model = MLEmbedModel(ndim=MAX_LEN, device=device)
    else:
        model = MetricModel()
        dataset = AudioOnlyDataset(audio_dir=AUDIO_DIR, split='test', device=device)
    data_loader = DataLoader(dataset, batch_size=8, num_workers=4)
    # model_path = "../train/result/ml_with_embed_train_data_epoch9_loss0.2000.pt"
    model = example_model_setting(model, args.model_path,args.model)
    example = next(iter(data_loader))

    model.to(device)

    with torch.no_grad():
        audio_embed = model.audio_to_embedding(example)

    # sentence_embedding = torch.mean(text_positive_embed, dim=0)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 5
    embeddings = []

    dataset = StoryTextDataset(TEXT_DIR, 'test', MAX_LEN)

    for i in range(5):
        with torch.no_grad():
            text = dataset.data[i]['recite_src']['text']
            tokenized = dataset.tokenizer.encode_plus(text,
                                                      add_special_tokens=True,
                                                      max_length=MAX_LEN,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_tensors='pt')

            embedding = model.text_to_embedding(tokenized['input_ids'].to(device),
                                                tokenized['attention_mask'].to(device))

            embedding = embedding.cpu()
            embeddings.append(embedding[0])

    embeddings = np.array(embeddings).squeeze(axis=1)
    print(embeddings.shape)

    audio_embed = audio_embed.cpu().numpy()
    cos_scores = cosine_similarity(audio_embed, embeddings)
    cos_scores = cos_scores.cpu().numpy()

    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", example['anchor'])
    print("\nTop 5 most similar sentences in corpus by emotion mapping:")

    for idx in top_results[0:top_k]:
        print(dataset.decode(dataset.data[idx]['recite_src']['text']), "(Score: %.4f)" % (cos_scores[idx]))


if __name__ == '__main__':
    main()
