import glob
import numpy as np
import gymu
from gymu.data import Composable
import gym_pygame
import jnu as J
import os

from torch.utils.data import DataLoader, IterableDataset
import webdataset as wb

def alone():
    env = gymu.make("Alone-v0")
    iterator = gymu.iterator(env, mode=gymu.mode.sard, max_length=500)
    gymu.data.write_episode(iterator, path="Alone-v0/ep-0")
    #gymu.data.write_episode(iterator, path="Alone-v0/ep-1.tar.gz")

    episodes = glob.glob("./Alone-v0/*.tar.gz")

    def print_worker(x):
        print(os.getpid())
        return x

   # dataset = wb.WebDataset(episodes)

    dataset = gymu.data.dataset(episodes).then(Composable.decode(keep_meta=True))
    #dataset = dataset.map(lambda x: (x['__key__'], x['__worker__']))
    dataset = dataset.shuffle(1000, initial=200)#.map(lambda x: (x['__key__'], x['__worker__'])).map(print_worker)

    loader = DataLoader(dataset, batch_size=8, num_workers=10)

    for x in loader:
        print(x['__key__'], x['__worker__'])

if __name__ == "__main__":
    alone()

    """
    class TestIterableDataset(IterableDataset):
        def __init__(self):
            self.x = np.arange(1000)
            self.i = 0

        def __iter__(self):
            return self
        
        def __next__(self):
            x = self.x[self.i], os.getpid()
            self.i += 1
            return x

    dataset = TestIterableDataset()
    loader = DataLoader(dataset, batch_size=8, num_workers=8)
    for x in loader:
        print(x)

    """




    
