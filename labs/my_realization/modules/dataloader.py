from sklearn.utils import shuffle
from math import ceil
class DataLoader(object):
    def __init__(self, X, y, batch_size=1, shuffle=False):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0

    def __len__(self) -> int:
        return ceil(self.num_samples() / self.batch_size)

    def num_samples(self) -> int:
        return self.X.shape[0]

    def __iter__(self):
        self.batch_id = 0
        if(self.shuffle):
            self.X, self.y = shuffle(self.X, self.y)
        return self

    def __next__(self):
        self.batch_id += 1
        if self.batch_id < len(self):
            return self.X[(self.batch_id - 1) * self.batch_size : self.batch_id * self.batch_size] \
                , self.y[(self.batch_id - 1) * self.batch_size : self.batch_id * self.batch_size]
        elif self.batch_id == len(self):
            return self.X[(self.batch_id - 1) * self.batch_size : ], self.y[(self.batch_id - 1) * self.batch_size : ]
        raise StopIteration
