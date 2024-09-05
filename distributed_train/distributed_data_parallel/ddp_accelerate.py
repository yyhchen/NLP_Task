import torch
import pandas as pd

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
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
        - 加入了 accelerate 后， DataLoader无需再加入 `sampler=DistributedSampler(trainset)` 参数，加入 `shuffle` 即可

        - 无需手工将模型通过 `LOCAL_RANK` 加入GPU, `model = DDP(model)`

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

    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = BertForSequenceClassification.from_pretrained(model_path)

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    """
        - 通过 accelerator.gather_for_metrics 管理 pred 与 refs; 使得后续无需手动 all_reduce 所有的 acc_num
    """
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, epoch=3, log_step=10):
    """
        通过 accelerator 管理反向传播计算的 loss; 将原生pytorch 的 `dist.all_reduce` 改为 `accelerator.reduce`
    """
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            optimizer.step()
            if global_step % log_step == 0:
                loss = accelerator.reduce(loss, "mean")
                accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate(model, validloader, accelerator)
        accelerator.print(f"ep: {ep}, acc: {acc}")


def main():
    # 初始化 accelerator 无需 dist.iinit_process_group(backend='nccl')
    accelerator = Accelerator()

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)

    train(model, optimizer, trainloader, validloader, accelerator)


if __name__ == "__main__":
    main()