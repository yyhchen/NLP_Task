# Distributed Training


---


## data parallel

DP 忽略，并不算真正意义上的分布式并行，而且并行策略对于现在的训练来说也不合适。


<br>
<br>



## distributed data parallel

- raw pytorch ddp
- pytorch + transformers ddp
- pytorch + accelerate ddp

### raw pytorch ddp
[ddp.py](/distributed_train/distributed_data_parallel/ddp.py) 使用原生 PyTorch 进行 DDP 训练

**执行方法:**

```bash
torchrun --nproc_per_node=2 ddp.py
```

<br>
<br>

### pytorch + transformers ddp
[ddp_transformers.py](/distributed_train/distributed_data_parallel/ddp_transformers.py) 结合 transformers库 和 PyTorch 打包成 trainer 进行 DDP 训练

**执行方法:**
```bash
python ddp_transformers.py
``` 
无需指定 `--nproc_per_node` 参数, 且无需使用 `torchrun` 启动也可使用多张显卡， transformers 帮我们做好了一切

**效果图如下：**
![ddp training](/assets/ddp_trainer_result.png)


<br>
<br>

### pytorch + accelerate ddp
[ddp_accelerate.py](/distributed_train/distributed_data_parallel/ddp_accelerate.py) 结合原生 PyTorch 和 accelerate库 进行 DDP 训练

**执行方法:**

方法一：
```bash
torchrun --nproc-per-node=2 ddp_accelerate.py
```

效果图如下:
![torchrun ddp_accelerate](/assets/ddp_accelerate_result.png)


方法二：
```bash
accelerate launch ddp_accelerate.py
```
同样的，accelerate 帮我们做完了全部的事情，无需指定 `--nproc-per-node` 参数启动即可 多卡训练.

**效果图如下:**
![accelerate ddp_accelerate](/assets/ddp_accelerate_result2.png)


当然，我们在方法二的效果图上可以看到有一些 `accelerate launch` 的一些警告，是关于 `accelerate config` 的内容，也就是需要指定 启动的配置参数，如果默认就会出现这些警告，可以通过 启动 `accelerate config` 配置内容消除警告


<br>
<br>


### accelerate advanced ddp

加入了 梯度累积，混合精度，实验记录，模型保存，断点续训的功能

[ddp_accelerate_advanced.py](/distributed_train/distributed_data_parallel/ddp_accelerate_advanced.py)

**执行方法:**

```bash
accelerate launch --mixed_precision bf16 ddp_accelerate_advanced.py 
```

**效果图如下:**
![accelerate ddp_accelerate](/assets/ddp_accelerate_advanced_result.png)

> tensorboard 功能可以通过 `command + shift + p` 输入 `tensorboard` 打开
