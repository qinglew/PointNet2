import h5py
import numpy as np
import torch
import torch.utils.data as Data


class ModelNet40(Data.Dataset):
    def __init__(self, dataset_dir, n=1024, split='train', data_augmentation=True):
        self.dataset_dir = dataset_dir
        txt_file = dataset_dir + '/{}_files.txt'.format(split)
        self.coordinates, self.labels = self.load_data(txt_file)
        self.n = n
        self.data_augmentation = data_augmentation
        print("ModelNet40 dataset {}ing split contains {} samples".format(split, len(self.coordinates)))

        self.category2code = {}
        self.code2category = []

        with open('../others/modelnet_id.txt', 'r') as f:
            for line in f:
                category, code = line.strip().split()
                self.category2code[category] = int(code)
                self.code2category.append(category)

        # the number of samples in every category
        self.categories_nums = []
        for i in range(len(np.unique(self.labels))):
            self.categories_nums.append(np.sum(self.labels == i))

    def load_data(self, txt_file):
        coordinates_list, labels_list = list(), list()
        with open(txt_file, 'r') as f:
            for line in f:
                data_file = line.strip().split('/')[-1]
                with h5py.File('/'.join([self.dataset_dir, data_file]), 'r') as h5:
                    coordinates = np.array(h5.get('data'))
                    labels = np.array(h5.get('label'))
                    coordinates_list.append(coordinates)
                    labels_list.append(labels)
        return np.concatenate(coordinates_list, axis=0), np.concatenate(labels_list, axis=0)

    def __getitem__(self, index):
        point_set, label = self.coordinates[index], self.labels[index]
        choice = np.random.choice(len(point_set), self.n, replace=True)
        point_cloud = point_set[choice, :]

        # center
        point_cloud = point_cloud - np.expand_dims(np.mean(point_cloud, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)), 0)
        point_cloud = point_cloud / dist

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_cloud[:, [0, 2]] = point_cloud[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)  # random jitter

        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))
        return point_cloud, label

    def __len__(self):
        return len(self.coordinates)


if __name__ == '__main__':
    dataset = ModelNet40(dataset_dir='data/modelnet40_ply_hdf5_2048', split='train')
    index = np.random.randint(0, len(dataset))
    sample, label = dataset[index]
    print(sample.shape)
    print(sample)
    print(label.shape)
    print(label)

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=10)
    data, labels = next(iter(data_loader))
    print(data.shape, labels.shape)

    print(dataset.categories_nums)
    print(np.sum(dataset.categories_nums))
