import numpy as np
from typing import Union, List
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from utils import whiten, normalize, svd_
from PIL import Image

# Synthetic data
class SyntheticDataset(Dataset):
    def __init__(self, samples: np.ndarray, labels: np.ndarray):
        super(SyntheticDataset, self).__init__()
        
        self.samples = samples # Data samples of shape (N, d)
        self.targets = labels # One-hot encoded labels of shape (N, K)
    
    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        sample = self.samples[idx, :]
        label = self.targets[idx, :]

        return sample, label
    

def generate_gmm_data(d: int,
                  K: int,
                  N_k: int,
                  data_rank: int = None,
                  nu: float = 1.0,
                  whiten_: bool = True,
                  normalize_: bool = False, 
                  batch_size: int = None,
                  shuffle: bool = False):
    
    normalize_ = normalize_ and not whiten_

    # Create data samples
    p = data_rank if data_rank is not None else d
    
    U = np.linalg.svd(np.random.randn(d, p), full_matrices=False).U
    mu_all = np.random.randn(K, p)
    
    X = np.array( [(np.sqrt(nu) * np.random.randn(N_k, p) + np.expand_dims(mu_all[k, :], axis=0)) @ U.T for k in range(K)] )
    X = np.reshape(X, (K * N_k, d)) # Train samples (N, d)
    
    # Pre-process if necessary
    if whiten_:
        X = whiten(X)
    if normalize_:
        X /= np.linalg.norm(X, axis=0)

    # Create labels
    y = np.repeat(np.arange(K), N_k, axis=0)
    Y = np.eye(K)[y] # Labels (N, K)
    

    # Create dataloader
    if batch_size is None:
        batch_size = X.shape[0]

    dataset = SyntheticDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, dataloader


def generate_uos_data(d: int,
                  K: int,
                  N_k: int,
                  subspace_rank: int = None,
                  nu: float = 1.0,
                  whiten_: bool = True,
                  normalize_: bool = False, 
                  batch_size: int = None,
                  shuffle: bool = False):
    
    normalize_ = normalize_ and not whiten_

    # Create data samples
    r = subspace_rank if subspace_rank is not None else d // K
    assert r <= d // 2
    
    U = svd_(np.random.randn(d, d))[0]
    U_all = [U[:, k*r:(k+1)*r] for k in range(K)]
    mu_all = np.zeros((K, d)) #np.random.randn(K, d)
    
    X = np.array( [np.random.randn(N_k, r) @ U_all[k].T + np.expand_dims(mu_all[k, :], axis=0) + np.sqrt(nu)*np.random.randn(N_k, d) for k in range(K)] )
    X = np.reshape(X, (K * N_k, d)) # Train samples (N, d)
    
    # Pre-process if necessary
    if whiten_:
        X = whiten(X)
    if normalize_:
        X /= np.linalg.norm(X, axis=0)

    # Create labels
    y = np.repeat(np.arange(K), N_k, axis=0)
    Y = np.eye(K)[y] # Labels (N, K)
    

    # Create dataloader
    if batch_size is None:
        batch_size = X.shape[0]

    dataset = SyntheticDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, dataloader

def generate_random_data(d: int,
                  K: int,
                  N_k: int,
                  data_rank: int = None,
                  whiten_: bool = True,
                  normalize_: bool = False, 
                  batch_size: int = None,
                  shuffle: bool = False):
    
    normalize_ = normalize_ and not whiten_
    
    # Create data samples
    p = data_rank if data_rank is not None else d
    
    N = N_k * K
    X = np.random.randn(N, d)
    
    if whiten_:
        X = whiten(X)
    if normalize_:
        X /= np.linalg.norm(X, axis=0)
    
    # Create labels via k-means
    k_means = KMeans(n_clusters=K)
    y = k_means.fit_predict(X)
    Y = np.eye(K)[y] # Labels (N, K)
    
    # create data loader
    if batch_size is None:
        batch_size = X.shape[0]
        
    dataset = SyntheticDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, dataloader




# Fashion MNIST
class PartialFashionMNIST(FashionMNIST):
    def __init__(self, root, K, N_k, one_hot=True, train=True, transform=None, download=True, **kwargs):
        super(PartialFashionMNIST, self).__init__(root=root, train=train, transform=transform, download=download, **kwargs)
        
        assert K > 0 and K <= 10
        if train:
            print("Using Fashion MNIST train set with {} classes!\n".format(K))
        else:
            print("Using Fashion MNIST val set with {} classes!\n".format(K))
        
        N = N_k * K
        self.targets = np.array(self.targets)
        all_data = []
        all_labels = []
        
        for k in range(K):
            idx = np.random.randint(low=0, high=np.count_nonzero(self.targets == k), size=(N_k,))
            all_data.append(self.data[self.targets == k][idx])
            all_labels.append(np.ones(N_k, dtype=np.uint8) * k)
        
        
        #perm = np.random.choice(np.arange(N), size=(N,), replace=False) 
        self.data = np.concatenate(all_data, axis=0) #[perm, ...] # Images of shape (N, height, width)
        self.targets = np.concatenate(all_labels) #[perm] # Size (N,) 
        if one_hot:
            self.targets = np.eye(K)[self.targets] # One-hot encoded labels of shape (N, K) 
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img.flatten(), target
            

def get_fashion_mnist(K: int, 
                      N_k: int, 
                      data_dir: str, 
                      batch_size: int = None, 
                      one_hot: bool = True, 
                      train: bool = True, 
                      shuffle: bool = False):
    
    if batch_size is None:
        batch_size = K * N_k
    
    # Transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.286, ), (0.353,))
    ])
    
    # Get dataset and dataloader
    fashion_mnist_dataset = PartialFashionMNIST(data_dir, K, N_k, one_hot=one_hot, train=train, transform=data_transform)
    fashion_mnist_loader = DataLoader(fashion_mnist_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return fashion_mnist_dataset, fashion_mnist_loader

