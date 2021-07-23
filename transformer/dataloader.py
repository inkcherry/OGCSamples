import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Seq2seq_Dataset(Dataset):
    def __init__(self, src_filepath,target_filepath):
        self.src_data = np.load(src_filepath, allow_pickle=True)
        self.trg_data = np.load(target_filepath, allow_pickle=True)

        start_token=np.zeros([self.trg_data.shape[0],1],dtype=np.int64)
        start_token.fill(711)

        self.trg_data=np.concatenate([start_token,self.trg_data],axis=1)
        # c=3
        # print("Pos data: {}".format(len(pos_data)))
        # print("Neg data: {}".format(len(neg_data)))
        # pos_label = np.array([1 for _ in pos_data])
        # neg_label = np.array([0 for _ in neg_data])
        # self.data = np.concatenate([pos_data, neg_data])
        # self.label = np.concatenate([pos_label, neg_label])

    def __len__(self):
        return len(self.src_data)
    def __getitem__(self, idx):
        src = torch.from_numpy(self.src_data[idx]).long()
        trg = torch.from_numpy(self.trg_data[idx]).long()
        # label = torch.nn.init.constant_(torch.zeros(1), int(self.label[idx])).long()
        return {"src": src, "trg": trg}


def seq2seq_dataloader(src_filepath,target_filepath,batch_size, shuffle, num_workers, pin_memory):
    dataset = Seq2seq_Dataset(src_filepath,target_filepath)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
