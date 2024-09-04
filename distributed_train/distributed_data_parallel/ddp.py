import os
import time
import torch
import pandas as pd
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "/root/private_data/models/hfl/chinese-macbert-large"

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def prepare_dataloader():
    """
        数据处理， sampler=DistributedSampler(trainset) 这是为了使得数据采样的时候不重叠，分布式训练需要这么处理数据
    """

    dataset = MyDataset()

    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))

    tokenizer = BertTokenizer.from_pretrained(model_path)

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(trainset))
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))

    return trainloader, validloader


def prepare_model_and_optimizer():

    """
        from torch.nn.parallel import DistributedDataParallel as DDP
    
        核心就是 model = DDP(model), 将模型copy到其他的GPU上
    """

    model = BertForSequenceClassification.from_pretrained(model_path)

    # "LOCAL_RANK" 这样的参数，是由代码动态获取的
    if torch.cuda.is_available():
        model = model.to(int(os.environ["LOCAL_RANK"]))

    model = DDP(model)

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


# 选择 一个 进程 进行信息打印
def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)


def evaluate(model, validloader):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    # 做一个信息 汇总
    dist.all_reduce(acc_num)
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, epoch=3, log_step=100):
    start_time = time.time()
    global_step = 0
    for ep in range(epoch):
        model.train()
        trainloader.sampler.set_epoch(ep)
        for batch in trainloader:
            if torch.cuda.is_available():
                # 将 tokenizer 后的 tensor 放进 GPU 
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        end_time = time.time()
        acc = evaluate(model, validloader)
        print_rank_0(f"ep: {ep}, acc: {acc}")
        print(f"total times: {end_time - start_time}")


def main():

    # 启动一个通信后端 (类似的还有MPI等)
    dist.init_process_group(backend="nccl")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    train(model, optimizer, trainloader, validloader)


if __name__ == "__main__":
    main()