import argparse
from tqdm import tqdm

from fusion.metric_learning import *
from config import *

from tensorboardX import SummaryWriter


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
        help='weight_decay'
    )

    parser.add_argument(
        '--cuda',
        default='cuda:0',
        help='cuda'
    )

    parser.add_argument(
        '--log_dir',
        default=metric_embedding_train_config['log_dir'],
        help='log_dir to save'
    )

    args = parser.parse_args()
    return args

args = parse_args()


def main():
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = SummaryWriter(log_dir=args.log_dir)

    AUDIO_DIR = metric_embedding_train_config['audio_dir']
    TEXT_DIR = metric_embedding_train_config['text_dir']
    MAX_LEN = metric_embedding_train_config['max_len']
    AUDIO_MAX = metric_embedding_train_config['audio_max']
    BATCH_SIZE = args.batch

    if args.train == True:
        train_dataloader = create_data_loader(AUDIO_DIR , TEXT_DIR, 'valid', MAX_LEN, AUDIO_MAX, BATCH_SIZE)
        valid_dataloader = create_data_loader(AUDIO_DIR , TEXT_DIR, 'test', MAX_LEN, AUDIO_MAX, BATCH_SIZE)

        print('data loading done')

        seed = 1024
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        device = args.cuda
        print('---------------------', device)
        model = MLEmbedModel(ndim=MAX_LEN, device=device)
        model = model.to(device)
        print('---config---')
        print(args)
        min_loss = float("inf")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            train(model, optimizer, train_dataloader, logger, epoch)
            loss, triplet_loss, triplet_distance_loss, cosine_similarity, manhattan_distance, euclidean_distance  = evaluate(model,valid_dataloader, logger, epoch)

            if min_loss > loss:
                temp = min_loss
                min_loss = loss
                if 'result' not in os.listdir():
                    os.mkdir('result')
                    torch.save(model,'../result/{}_epoch{}.pt'.format('metric_with_embed', epoch))
                    print("-"*10,"Saving Model - loss {:.4f} ->  {:.4f}".format(temp, min_loss),"-"*10)

    logger.flush()


def train(model, optimizer, dataloader, logger, epoch):
    print("Train start")
    model.train()
    tqdm_train = tqdm(total=len(dataloader), position=1)

    loss_per_epoch = []
    for idx, batch in enumerate(dataloader):
        _, _, _, loss = model(batch)
        logger.add_scalar('train_loss', loss, idx)
        loss_per_epoch.append(loss)

        tqdm_train.set_description('loss is {:.2f}'.format(loss.item()))
        tqdm_train.update()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tqdm_train.close()
    logger.add_scalar('train_loss_per_epoch', sum(loss_per_epoch) / len(loss_per_epoch), epoch)


def evaluate(model,valid_dataloader, logger, epoch):
    print("Eval start")
    model.eval()
    tqdm_valid = tqdm(total=len(valid_dataloader), position=1)

    with torch.no_grad():
        losses = 0
        triplet_losses = 0
        triplet_distance_losses = 0
        cosine_similarities = 0
        manhattan_distances = 0
        euclidean_distances = 0

        for idx, batch in tqdm(enumerate(valid_dataloader)):
            score = model.evaluate(batch)

            loss = score['loss'].item()
            losses += loss
            logger.add_scalar('valid/loss', loss, idx)

            tqdm_valid.set_description('loss is {:.2f}'.format(loss))
            tqdm_valid.update()

            triplet_loss = score['triplet_loss'].item()
            triplet_losses += triplet_loss
            logger.add_scalar('valid/triplet_loss', triplet_loss, idx)

            triplet_distance_loss = score['triplet_distance_loss'].item()
            triplet_distance_losses += triplet_distance_loss
            logger.add_scalar('valid/triplet_distance_loss', triplet_distance_loss, idx)

            cosine_similarity = score['cosine_similarity'].item()
            cosine_similarities += cosine_similarity
            logger.add_scalar('valid/cosine_similarity', cosine_similarity, idx)

            manhattan_distance = score['manhattan_distance'].item()
            manhattan_distances += manhattan_distance
            logger.add_scalar('valid/manhattan_distance', manhattan_distance, idx)

            euclidean_distance = score['euclidean_distance'].item()
            euclidean_distances += euclidean_distance
            logger.add_scalar('valid/euclidean_distance', euclidean_distance, idx)

        losses = losses/len(valid_dataloader)
        triplet_losses = triplet_losses / len(valid_dataloader)
        triplet_distance_losses = triplet_distance_losses / len(valid_dataloader)
        cosine_similarities = cosine_similarities / len(valid_dataloader)
        manhattan_distances = manhattan_distances / len(valid_dataloader)
        euclidean_distances = euclidean_distances / len(valid_dataloader)

        logger.add_scalar('valid/loss_per_epoch', losses, epoch)
        logger.add_scalar("valid/triplet_loss_per_epoch", triplet_losses, epoch)
        logger.add_scalar("valid/triplet_distance_loss_per_epoch", triplet_distance_losses, epoch)
        logger.add_scalar("valid/cosine_similarity_per_epoch", cosine_similarities, epoch)
        logger.add_scalar("valid/manhattan_distance_per_epoch", manhattan_distances, epoch)
        logger.add_scalar("valid/euclidean_distance_per_epoch", euclidean_distances, epoch)

        tqdm_valid.close()

        print('Validation Result: Loss - {:.4f} | triplet_loss - {:.3f} |\
              triplet_distance_loss - {:.3f} | cosine_similarity = {:.3f} | manhattan_distance - {:.3f} | \
              euclidean_distancee - {:.3f}'.format(losses, triplet_losses, triplet_distance_losses, cosine_similarities, manhattan_distances, euclidean_distances))

    return losses, triplet_losses, triplet_distance_losses, cosine_similarities, manhattan_distances, euclidean_distances


if __name__ == '__main__':
    main()
