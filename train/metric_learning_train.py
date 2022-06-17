import argparse
from tqdm import tqdm

from fusion.metric_learning import *
from config import *


def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')
    parser.add_argument(
        '--train',
        default=True,
        required=False,
        help='run train'
    )
    parser.add_argument(
        '--epochs',
        default=metric_embedding_train_config['epochs'],
        type=int,
        required=False,
        help='epochs'
    )
    parser.add_argument(
        '--batch',
        default=metric_embedding_train_config['batch_size'],
        type=int,
        required=False,
        help='batch size'
    )
    parser.add_argument(
        '--lr',
        default=metric_embedding_train_config['lr'],
        type=float,
        required=False,
        help='learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        default=metric_embedding_train_config['weight_decay'],
        type=float,
        required=False,
        help='learning rate'
    )

    parser.add_argument(
        '--cuda',
        default='cuda:0',
        help='class weight'
    )
    args = parser.parse_args()
    return args

args = parse_args()


def main():
    AUDIO_DIR = metric_embedding_train_config['audio_dir']
    TEXT_DIR = metric_embedding_train_config['text_dir']
    MAX_LEN = metric_embedding_train_config['max_len']
    AUDIO_MAX = metric_embedding_train_config['audio_max']
    BATCH_SIZE = args.batch

    if args.train == True:
        train_dataloader = create_data_loader(AUDIO_DIR , TEXT_DIR, 'train', MAX_LEN, AUDIO_MAX, BATCH_SIZE)
        valid_dataloader = create_data_loader(AUDIO_DIR , TEXT_DIR, 'valid', MAX_LEN, AUDIO_MAX, BATCH_SIZE)

        seed = 1024
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = MLEmbedModel(ndim=MAX_LEN)
        device = args.cuda
        print('---------------------', device)
        model = model.to(device)
        print('---config---')
        print(args)
        min_loss = float("inf")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            train(model, optimizer, train_dataloader)
            loss, triplet_loss, triplet_distance_loss, cosine_similarity, manhattan_distance, euclidean_distance  = evaluate(model,valid_dataloader)

            if min_loss > loss:
                temp = min_loss
                min_loss = loss
                if 'result' not in os.listdir():
                    os.mkdir('result')
                    torch.save(model,'./result/{}_epoch{}.pt'.format(args.model_name,epoch))
                    print("-"*10,"Saving Model - loss {:.4f} ->  {:.4f}".format(temp, min_loss),"-"*10)


def train(model, optimizer, dataloader):
    print("Train start")
    model.train()
    tqdm_train = tqdm(total=len(dataloader), position=1)

    for batch in enumerate(dataloader):
        _, _, _, loss = model(batch)
        tqdm_train.set_description('loss is {:.2f}'.format(loss.item()))
        tqdm_train.update()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tqdm_train.close()


def evaluate(model,valid_dataloader):
    print("Eval start")
    model.eval()

    with torch.no_grad():
        losses = 0
        triplet_losses = 0
        triplet_distance_losses = 0
        cosine_similarities = 0,
        manhattan_distances = 0,
        euclidean_distances = 0

        for batch in valid_dataloader:
            score = model.evaulate(batch)

            loss = score['loss'].item()
            losses += loss

            triplet_loss = score['triplet_loss'].item()
            triplet_losses += triplet_loss

            triplet_distance_loss = score['triplet_distance_loss'].item()
            triplet_distance_losses += triplet_distance_loss

            cosine_similarity = score['cosine_similarity'].item()
            cosine_similarities += cosine_similarity

            manhattan_distance = score['manhattan_distance'].item()
            manhattan_distances += manhattan_distance

            euclidean_distance = score['euclidean_distance'].item()
            euclidean_distances += euclidean_distance

        losses = losses/len(valid_dataloader)
        triplet_losses = triplet_losses / len(valid_dataloader)
        triplet_distance_losses = triplet_distance_losses / len(valid_dataloader)
        cosine_similarities = cosine_similarities / len(valid_dataloader)
        manhattan_distances = manhattan_distances / len(valid_dataloader)
        euclidean_distances = euclidean_distances / len(valid_dataloader)



        print('Validation Result: Loss - {:.4f} | triplet_loss - {:.3f} |\
              triplet_distance_loss - {:.3f} | cosine_similarity = {:.3f} | manhattan_distance - {:.3f} | \
              euclidean_distancee - {:.3f}'.format(losses, triplet_losses, triplet_distance_losses, cosine_similarities, manhattan_distances, euclidean_distances))

    return losses, triplet_losses, triplet_distance_losses, cosine_similarities, manhattan_distances, euclidean_distances


if __name__ == '__main__':
    main()
