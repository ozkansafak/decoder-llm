import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
os.environ['MASTER_ADDR'] = 'localhost' ### the IP of vm_gpu02
os.environ['MASTER_PORT'] = '9000'


class Model(nn.Module):
    # Our model
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Conv2d(1,10,3)
        self.bn1 = nn.BatchNorm2d(10)
        self.fc2= nn.Conv2d(10,20,3)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc3= nn.Linear(11520,10)
    def forward(self,x):
        print(f'input_size: {x.size()}')
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = x.view(x.size(0),-1)
        x = self.fc3(x)
        print(f'output_size: {x.size()}')
        return(x)


def main(rank):
    print(f"rank_{rank}: in train() ")
    init_process_group(backend = 'nccl',
                       init_method = 'env://',
                       world_size=8,
                       rank=rank)
    
    torch.manual_seed(0)
    model = Model()
    print(f"rank_{rank}: model loaded")
    torch.cuda.set_device(rank)
    model = model.to(rank)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lr_sch = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(rank)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids = [rank])
    
    mnist = torchvision.datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.ToTensor())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(mnist,
                                                                    num_replicas=4,
                                                                    rank=rank)

    dataloader = DataLoader(mnist, batch_size=32, num_workers=4, pin_memory=True,
                            sampler=train_sampler)

    t_loss = None
    for epoch in range(2):
        total_loss =0
        for X,y in dataloader:
            X = X.to(rank)
            y = y.long().to(rank)
            pred = model(X)
            loss = criterion(pred, y)
            t_loss= loss.item() if t_loss is None else t_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Loss: {t_loss/len(dataloader)}')

    destroy_process_group()
    
    
if __name__=='__main__':
    mp.spawn(main, nprocs=8)




