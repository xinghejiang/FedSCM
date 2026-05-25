import os
import torch
import random
import logging
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = ['uid', 'iid']
        feature_tensors = [torch.tensor(data[col].values, dtype=torch.long) for col in self.features]
        self.feature_matrix = torch.stack(feature_tensors, dim=1)
        self.labels = torch.tensor(data['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.feature_matrix[idx], self.labels[idx]


class MovieLensDataLoader:
    def __init__(self, data_dir, num_negatives, batch_size, num_users, num_items, dataset_name):
        self.data_dir = data_dir
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.num_users = num_users
        self.num_items = num_items
        self.dataset_name = dataset_name

        df = self._load_raw_data(self.dataset_name, self.num_users)
        self.item_pool = set(df['iid'].unique())
        self.user_pool = set(df['uid'].unique())
        self.negatives = self._sample_negative(df)
        self.train_data, self.vali_data, self.test_data = self._split_loo(df)

    def _load_raw_data(self, dataset_name, num_users):
        rating_file = os.path.join(self.data_dir, 'ratings.dat')
        sep = '::' if dataset_name == 'ml-1m' else ','
        df = pd.read_csv(rating_file, sep=sep, header=None,
                         names=['uid', 'iid', 'rating', 'timestamp'], engine='python')
        logging.info(f'Loaded {len(df)} interactions from {rating_file}')
        df = df[df['rating'] > 0]
        df['rating'] = (df['rating'] > 0).astype(float)
        user_id_mapping = {uid: idx for idx, uid in enumerate(df['uid'].unique())}
        df['uid'] = df['uid'].map(user_id_mapping)
        item_id_mapping = {iid: idx for idx, iid in enumerate(df['iid'].unique())}
        df['iid'] = df['iid'].map(item_id_mapping)
        df = df[df['uid'] < num_users]
        return df

    def _sample_negative(self, df):
        interact_status = df.groupby('uid')['iid'].apply(set).reset_index().rename(columns={'iid': 'interacted_items'})

        def sample_for_user(interacted_items):
            all_neg = self.item_pool - interacted_items
            neg_samples = random.sample(list(all_neg), min(99, len(all_neg)))
            neg_items = all_neg - set(neg_samples)
            return neg_items, neg_samples

        result = interact_status['interacted_items'].apply(sample_for_user)
        interact_status['negative_items'] = result.apply(lambda x: x[0])
        interact_status['negative_samples'] = result.apply(lambda x: x[1])
        return interact_status[['uid', 'negative_items', 'negative_samples']]

    def _split_loo(self, df):
        df['rank_latest'] = df.groupby(['uid'])['timestamp'].rank(method='first', ascending=False)
        test = df[df['rank_latest'] == 1]
        val = df[df['rank_latest'] == 2]
        train = df[df['rank_latest'] > 2]
        assert train['uid'].nunique() == test['uid'].nunique() == val['uid'].nunique()
        return train[['uid', 'iid', 'rating']], val[['uid', 'iid', 'rating']], test[['uid', 'iid', 'rating']]

    def _prepare_test_data(self, df):
        test_ratings = pd.merge(df, self.negatives[['uid', 'negative_samples']], on='uid')
        samples = []
        for uid, user_data in test_ratings.groupby('uid'):
            for row in user_data.itertuples():
                samples.append([row.uid, row.iid, row.rating])
                for neg_item in row.negative_samples:
                    samples.append([row.uid, neg_item, 0.0])
        samples = pd.DataFrame(samples, columns=['uid', 'iid', 'rating'])
        return {uid: DataLoader(MovieLensDataset(g), batch_size=self.batch_size, shuffle=False)
                for uid, g in samples.groupby('uid')}

    def _prepare_vali_data(self, df):
        vali_ratings = pd.merge(df, self.negatives[['uid', 'negative_samples']], on='uid')
        samples = []
        for uid, user_data in vali_ratings.groupby('uid'):
            for row in user_data.itertuples():
                samples.append([row.uid, row.iid, row.rating])
                for neg_item in row.negative_samples:
                    samples.append([row.uid, neg_item, 0.0])
        samples = pd.DataFrame(samples, columns=['uid', 'iid', 'rating'])
        return {uid: DataLoader(MovieLensDataset(g), batch_size=self.batch_size, shuffle=False)
                for uid, g in samples.groupby('uid')}

    @property
    def get_train_data(self):
        return self.train_data

    @property
    def get_test_data(self):
        return self._prepare_test_data(self.test_data)

    @property
    def get_vali_data(self):
        return self._prepare_vali_data(self.vali_data)

    def negative_sampling(self, num_negatives):
        train_ratings = pd.merge(self.train_data, self.negatives[['uid', 'negative_items']], on='uid')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(
            lambda x: random.sample(list(x) if isinstance(x, set) else x, min(num_negatives, len(x))))
        samples = []
        for uid, user_data in train_ratings.groupby('uid'):
            for row in user_data.itertuples():
                samples.append([row.uid, row.iid, row.rating])
                for neg_item in row.negatives:
                    samples.append([row.uid, neg_item, 0.0])
        samples = pd.DataFrame(samples, columns=['uid', 'iid', 'rating'])
        return {uid: DataLoader(MovieLensDataset(g), batch_size=self.batch_size, shuffle=True)
                for uid, g in samples.groupby('uid')}
