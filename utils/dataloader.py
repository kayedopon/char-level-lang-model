from .dataset import Dataset

import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size=64, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle == True:
                np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            X_batch, y_batch = self.dataset[start:start+self.batch_size]
            
            yield np.array(X_batch), np.array(y_batch)

    def __len__(self):
         return int(np.ceil(len(self.dataset) / self.batch_size))

            

### Came up with better implementaion
    # def __iter__(self):
    #     batch_num = len(self.dataset) / self.batch_size
    #     current = 0

    #     for i in range(int(batch_num)):
    #         batched_X = self.dataset.X[current:current+self.batch_size]
    #         batched_y = self.dataset.X[current:current+self.batch_size]
    #         current += 64
    # 
    #         yield batched_X, batched_y

    #     if batch_num > int(batch_num):
    #         extra = len(self.dataset) - (int(batch_num) * self.batch_size)
    #         batched_X = self.dataset.X[current:current+extra]
    #         batched_y = self.dataset.y[current:current+extra]
    # 
    #         yield batched_X, batched_y