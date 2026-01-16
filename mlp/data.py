import numpy as np
from typing import Union, List
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from utils import whiten, normalize, orthogonalize, svd_
from PIL import Image

from scipy.stats import ortho_group

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
                  data_scale: float = 1.0,
                  nu: float = 1.0,
                  whiten_: bool = False,
                  normalize_: bool = False, 
                  batch_size: int = None,
                  shuffle: bool = False):
    
    normalize_ = normalize_ and not whiten_
    
    N = N_k * K
    
    # Create data samples
    mu_all = np.random.randn(K, d) / np.sqrt(d)
    
    X = np.array( [np.expand_dims(mu_all[k, :], axis=0) + ( nu * np.random.randn(N_k, d) / np.sqrt(d) ) for k in range(K)] )
    X = np.reshape(X, (N, d)) # Train samples (N, d)
    
    #X = np.random.randn(N, d)
    X /= np.expand_dims(np.linalg.norm(X, axis=1), 1)
    
    # Pre-process if necessary
    if whiten_:
        X = data_scale * whiten(X)
    elif normalize_:
        X = normalize(X)

    # Create labels
    y = np.repeat(np.arange(K), N_k, axis=0)
    Y = np.eye(K)[y] # Labels (N, K)
    

    # Create dataloader
    if batch_size is None:
        batch_size = X.shape[0]

    dataset = SyntheticDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    grad_dataloader = DataLoader(dataset, batch_size=N, shuffle=shuffle)
    
    return dataset, dataloader, grad_dataloader


def generate_random_data(d: int,
                  K: int,
                  N_k: int,
                  kappa_X: float = 1.0,
                  orthogonalize_: bool = False,
                  normalize_: bool = False, 
                  batch_size: int = None,
                  shuffle: bool = False):
    
    normalize_ = normalize_ and not orthogonalize_

    
    N = N_k * K
    X = np.random.randn(N, d) / np.sqrt(d)
    #Q = ortho_group.rvs(d)[:, :N]
    #s = np.ones(N)
    
    #assert kappa_X >= 1
    
    #s[-1] = 1 / kappa_X
    #X = ortho_group.rvs(N) @ np.diag(s) @ Q.T
    
    
    if orthogonalize_:
        X = orthogonalize(X) + ( (1 / N) * np.random.randn(*X.shape) )
    elif normalize_:
        X = normalize(X)
    
    # Create labels via k-means
#     k_means = KMeans(n_clusters=K)
#     y = k_means.fit_predict(X)
#     Y = np.eye(K)[y] # Labels (N, K)
    
    y = np.repeat(np.arange(K), N_k, axis=0)
    Y = np.eye(K)[y] # Labels (N, K)
    
    # create data loader
    if batch_size is None:
        batch_size = X.shape[0]
        
    dataset = SyntheticDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    grad_dataloader = DataLoader(dataset, batch_size=N, shuffle=shuffle)
    
    return dataset, dataloader, grad_dataloader




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

