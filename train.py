import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse
from tqdm import tqdm

from tensorboardX import SummaryWriter

from dataset.multi import ThreeMultiDataset, TwoMultiDataset
from model.multi import ThreeBranchMetricModel, TwoBranchMetricModel

def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', type=str, default='data/Story_dataset/')
    parser.add_argument('--audio_dir', type=str, default='data/MTG/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='ThreeBranch', help= "ThreeBranch or TwoBranch")
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--ckp_per_step', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default='result/tensorboard')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    if args.model == "ThreeBranch":
        train_dataset = ThreeMultiDataset(audio_dir=args.audio_dir, text_dir=args.text_dir, split='train', device=device)
        valid_dataset = ThreeMultiDataset(audio_dir=args.audio_dir, text_dir=args.text_dir, split='valid', device=device)
    elif args.model == "TwoBranch":
        train_dataset = TwoMultiDataset(audio_dir=args.audio_dir, text_dir=args.text_dir, split='train')
        valid_dataset = TwoMultiDataset(audio_dir=args.audio_dir, text_dir=args.text_dir, split='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    print('data loading done')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = SummaryWriter(log_dir=args.log_dir)

    # model
    if args.model == "ThreeBranch":
        model = ThreeBranchMetricModel()
    elif args.model == "TwoBranch":
        model = TwoBranchMetricModel()

    # optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    if device == 'cuda':
        model = model.cuda(device)

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model = model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'{start_epoch}-th model loaded')

    for epoch in range(start_epoch, args.epoch):
        train(model, train_dataloader,optimizer,logger, epoch)
        val_loss, triplet_losses, triplet_distance_losses, \
        cosine_similarities, manhattan_distances, euclidean_distances = validate(model, valid_dataloader, logger, epoch)

        print(f'{epoch} done')
        print('start saving checkpoint')
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.cheeckpoint_dir)
        ckp_path = os.path.join(args.checkpoint_dir, f'metric_{epoch}_val_loss_{val_loss:.5f}.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckp_path)

    logger.flush()


def train(model, dataloader, optimizer, logger, epoch):  # for one epoch
    print("Train start")
    model.train()
    tqdm_train = tqdm(total=len(dataloader), position=1)

    loss_per_epoch = []
    for idx, batch in enumerate(dataloader):
        loss = model(batch)
        logger.add_scalar('train_loss', loss, idx)
        loss_per_epoch.append(loss)
        tqdm_train.set_description('loss is {:.2f}'.format(loss.item()))

        tqdm_train.update()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tqdm_train.close()
    logger.add_scalar('train_loss_per_epoch', sum(loss_per_epoch) / len(loss_per_epoch), epoch)


def validate(model, valid_dataloader, logger, epoch):
    print("Eval start")
    model.eval()

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
            manhattan_distances -= manhattan_distance
            logger.add_scalar('valid/manhattan_distance', manhattan_distance, idx)

            euclidean_distance = score['euclidean_distance'].item()
            euclidean_distances -= euclidean_distance
            logger.add_scalar('valid/euclidean_distance', euclidean_distance, idx)

        losses = losses / len(valid_dataloader)
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

        print('Validation Result: Loss - {:.4f} | triplet_loss - {:.3f} |\
              triplet_distance_loss - {:.3f} | cosine_similarity = {:.3f} | manhattan_distance - {:.3f} | \
              euclidean_distancee - {:.3f}'.format(losses, triplet_losses, triplet_distance_losses, cosine_similarities,
                                                   manhattan_distances, euclidean_distances))

    return losses, triplet_losses, triplet_distance_losses, cosine_similarities, manhattan_distances, euclidean_distances


if __name__ == '__main__':
    main()
