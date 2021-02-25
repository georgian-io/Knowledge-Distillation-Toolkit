import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F

def postprocess_features(feats, sample_rate):
    if feats.dim() == 2: feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    with torch.no_grad():
        feats = F.layer_norm(feats, feats.shape)
    return feats

def get_feature(batch_sample):
    return postprocess_features(batch_sample[0][0], batch_sample[1])

def get_padding_mask(batch_sample):
    return torch.BoolTensor(batch_sample[0].size(1)).fill_(False)

def get_batch_encoder_input(batch_samples):
    features = [get_feature(batch_sample) for batch_sample in batch_samples]
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    padding_masks = [get_padding_mask(batch_sample) for batch_sample in batch_samples]
    padding_masks = torch.nn.utils.rnn.pad_sequence(padding_masks, batch_first=True, padding_value=True)
    mask = False
    features_only = True
    return features, padding_masks, mask, features_only

class LibriSpeechDataLoader:

    """
    Data loaders for the LibriSpeech dataset.

    Arguments:
        train_batch_size (int): batch size for the training data loader
        val_batch_size (int): batch size for the validation data loader
        num_workers (int): number of workers for training and validation data loaders
        train_data_path (str): Path to training data
        val_data_path (str): Path to validation data
        train_on_dev_clean (bool): Set to True if you want to train on parts of the dev-clean dataset and validate on the other part. This is useful when testing ideas
        use_train_clean_100 (bool): Set to True if using LibriSpeech's train-clean-100 dataset during training
        use_train_clean_360 (bool): Set to True if using LibriSpeech's train-clean-360 dataset during training
        use_train_other_500 (bool): Set to True if using LibriSpeech's train-other-500 dataset during training
    """

    def __init__(self,
                 train_batch_size,
                 val_batch_size,
                 num_workers,
                 train_data_path,
                 val_data_path,
                 train_on_dev_clean,
                 use_train_clean_100,
                 use_train_clean_360,
                 use_train_other_500,
                 ):

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        dev_clean_dataset = torchaudio.datasets.LIBRISPEECH(val_data_path, url='dev-clean', download=False)
        dev_other_dataset = torchaudio.datasets.LIBRISPEECH(val_data_path, url='dev-other', download=False)
        dev_other_data_loader = DataLoader(dev_other_dataset,
                                           batch_size = val_batch_size,
                                           shuffle = False,
                                           num_workers = num_workers)

        if train_on_dev_clean:
            train_data_loader, dev_train_data_loader, dev_clean_data_loader = self.create_data_loaders_from_dev_clean(dev_clean_dataset,
                                                                                                                      train_batch_size,
                                                                                                                      val_batch_size,
                                                                                                                      num_workers)
        else:
            train_data_loader, dev_train_data_loader = self.create_data_loaders_from_train_dataset(train_data_path,
                                                                                                   train_batch_size,
                                                                                                   val_batch_size,
                                                                                                   num_workers,
                                                                                                   use_train_clean_100,
                                                                                                   use_train_clean_360,
                                                                                                   use_train_other_500,)
            dev_clean_data_loader = DataLoader(dev_clean_dataset,
                                               batch_size = val_batch_size,
                                               shuffle = False,
                                               num_workers = num_workers)

        self.train_data_loader = train_data_loader
        self.val_data_loaders = {
                                 #"dev_train": dev_train_data_loader,
                                 "dev_clean": dev_clean_data_loader,
                                 #"dev_other": dev_other_data_loader
                                }

    def create_data_loaders_from_dev_clean(self,
                                           dev_clean_dataset,
                                           train_batch_size,
                                           val_batch_size,
                                           num_workers):

        """
        Create train_data_loader and dev_train_data_loader from dev_clean_dataset.
        Parts of dev_clean_dataset will be used for training, and the other part will be used for validating.

        Arguments:
            dev_clean_dataset (torchaudio.datasets.LIBRISPEECH): dev-clean data set from LibriSpeech
            train_batch_size (int): batch size for the training data loader
            val_batch_size (int): batch size for the validation data loader
            num_workers (int): number of workers for the data loaders

        Returns:
            train_data_loader (torch.utils.data.DataLoader): data loader for training created from the dev clean dataset
            dev_train_data_loader (torch.utils.data.DataLoader): data loader for validating created from the dev clean dataset
        """

        train_dataset, val_dataset = torch.utils.data.random_split(dev_clean_dataset,
                                                                   [2203,500],
                                                                   generator=torch.Generator().manual_seed(42))

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=train_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        collate_fn=get_batch_encoder_input)
        dev_train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=val_batch_size,
                                                            shuffle=False,
                                                            num_workers=num_workers,
                                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(torch.randint(high=2203, size=(500,))),)
        dev_clean_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                            batch_size=val_batch_size,
                                                            shuffle=False,
                                                            num_workers=num_workers)
        return train_data_loader, dev_train_data_loader, dev_clean_data_loader


    def create_data_loaders_from_train_dataset(self,
                                               train_data_path,
                                               train_batch_size,
                                               val_batch_size,
                                               num_workers,
                                               use_train_clean_100,
                                               use_train_clean_360,
                                               use_train_other_500):
        """
        Create train_data_loader and dev_train_data_loader from training datasets of LibriSpeech.
        Create the joint training dataset based on user's selections.

        Arguments:
            train_data_path (str): path to LibriSpeech training data
            train_batch_size (int): batch size for train_data_loader
            val_batch_size (int): batch size for dev_traiin_data_loader
            num_workers (int): number of workers for data loaders
            use_train_clean_100 (bool): Set to True if using LibriSpeech's train-clean-100 dataset during training
            use_train_clean_360 (bool): Set to True if using LibriSpeech's train-clean-360 dataset during training
            use_train_other_500 (bool): Set to True if using LibriSpeech's train-other-500 dataset during training

        Returns:
            train_data_loader (torch.utils.data.DataLoader): data loader for training created from LibriSpeech training datasets
            dev_train_data_loader (torch.utils.data.DataLoader): data loader for validating created from LibriSpeech training datasets
        """
        selected_datasets = []
        if use_train_clean_100: selected_datasets.append(torchaudio.datasets.LIBRISPEECH(train_data_path, url='train-clean-100', download=False))
        if use_train_clean_360: selected_datasets.append(torchaudio.datasets.LIBRISPEECH(train_data_path, url='train-clean-360', download=False))
        if use_train_other_500: selected_datasets.append(torchaudio.datasets.LIBRISPEECH(train_data_path, url='train-other-500', download=False))
        train_dataset = torch.utils.data.ConcatDataset(selected_datasets)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=train_batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        collate_fn=get_batch_encoder_input)
        dev_train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=val_batch_size,
                                                            shuffle=False,
                                                            num_workers=num_workers,
                                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(torch.randint(high=len(train_data_loader), size=(2000,))),)
        return train_data_loader, dev_train_data_loader

    def get_train_data_loader(self):
        return self.train_data_loader

    def get_val_data_loaders(self):
        return self.val_data_loaders
