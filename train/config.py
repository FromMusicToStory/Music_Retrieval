metric_embedding_train_config = {
    'audio_dir': 'dataset/mtg-jamendo-dataset/',
    'text_dir': 'dataset/Story_dataset/',
    'max_len': 512,
    'audio_max': 500,
    'epochs': 250,
    'batch_size': 8,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'accumulation_steps': 8,
    'cuda': 'cuda:0',
    'log_dir': 'train/result/tensorboard',
    'model_name': 'ml_with_embed'
}