import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image



# CIFAR 10
class PartialCIFAR10(CIFAR10):
    def __init__(self, root, K, N_k, one_hot=True, train=True, transform=None, download=True, permute=True, **kwargs):
        super().__init__(root=root, train=train, transform=transform, download=download, **kwargs)

        assert K > 0 and K <= 10
        if train:
            print("Using CIFAR10 train set with {} classes!\n".format(K))
        else:
            print("Using CIFAR10 val set with {} classes!\n".format(K))

        # Get random images from first K classes
        self.targets = np.array(self.targets)
        all_data = []
        all_labels = []
        
        for k in range(K):
            idx = np.random.randint(low=0, high=np.count_nonzero(self.targets == k), size=(N_k,))
            all_data.append(self.data[self.targets == k][idx])
            all_labels.append(np.ones(N_k, dtype=np.uint8) * k)


        # Combine all data into one
        self.data = np.concatenate(all_data, axis=0)
        self.targets = np.concatenate(all_labels)
        if one_hot:
            self.targets = np.eye(K)[self.targets]
        
        
        # Randomly shuffle data points and targets
        if permute: 
            perm = np.random.choice(np.arange(N_k * K), replace=False, size=N_k * K)
            self.data = self.data[perm, ...]
            self.targets = self.targets[perm, ...]

            
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target


def get_cifar10(K: int, 
                N_k: int, 
                data_dir: str, 
                batch_size: int = None, 
                one_hot: bool = True,
                train: bool = True, 
                shuffle: bool = True):
    
    if batch_size is None:
        batch_size = K * N_k
    
    # Transform
    transforms_list = [transforms.Resize((224, 224)), 
                       transforms.ToTensor(), 
                       transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                       ]
    data_transform = transforms.Compose(transforms_list)
    
    # Get dataset and dataloader
    cifar10_dataset = PartialCIFAR10(data_dir, K, N_k, one_hot=one_hot, train=train, transform=data_transform)
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=shuffle) #, num_workers=1)

    return cifar10_dataset, cifar10_loader


# CIFAR-100 
class PartialCIFAR100(CIFAR100):
    def __init__(self, root, K, N_k, one_hot=True, train=True, transform=None, download=True, **kwargs):
        super().__init__(root=root, train=train, transform=transform, download=download, **kwargs)

        assert K > 0 and K <= 100
        if train:
            print("Using CIFAR100 train set with {} classes!\n".format(K))
        else:
            print("Using CIFAR100 val set with {} classes!\n".format(K))

        self.targets = np.array(self.targets)
        all_data = []
        all_labels = []
        
        for k in range(K):
            idx = np.random.randint(low=0, high=np.count_nonzero(self.targets == k), size=(N_k,))
            all_data.append(self.data[self.targets == k][idx])
            all_labels.append(np.ones(N_k, dtype=np.uint8) * k)

        self.data = np.concatenate(all_data, axis=0)
        self.targets = np.concatenate(all_labels, dtype=np.uint8)
        if one_hot:
            self.targets = np.eye(K)[self.targets]
            
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

def get_cifar100(K: int, 
                N_k: int, 
                data_dir: str, 
                batch_size: int = None, 
                one_hot: bool = True,
                train: bool = True, 
                shuffle: bool = True):
    
    if batch_size is None:
        batch_size = K * N_k
    
    # Transform
    transforms_list = [transforms.Resize((224, 224)), 
                       transforms.ToTensor(), 
                       transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
                       ]

    data_transform = transforms.Compose(transforms_list)
    
    # Get dataset and dataloader
    cifar100_dataset = PartialCIFAR100(data_dir, K, N_k, one_hot=one_hot, train=train, transform=data_transform)
    cifar100_loader = DataLoader(cifar100_dataset, batch_size=batch_size, shuffle=shuffle) #, num_workers=1)

    return cifar100_dataset, cifar100_loader