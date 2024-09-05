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



<br>
<br>



### DeepSpeed + accelerate 

在 `accelerate` 中使用 `DeepSpeed` 集成，详情可参考[官方文档](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)

训练代码用的仍然是 [ddp_accelerate_advanced.py](/distributed_train/DeepSpeed/ddp_accelerate_advanced.py)


**执行方法:**

先配置 `accelerate config`:
![accelerate config by use deepspeed](/assets/accelerate_config_deepspeed.png)

> 在上图中，将 **DeepSpeed's ZeRO optimization stage** 选择为 2, 是因为2比3的速度更快, 前提是一张卡能放入整个模型的情况下，选择 3 会拆分模型，增加通信时间延长模型训练时间.

将配置生成的 [`default_config.yaml`](/distributed_train/DeepSpeed/default_config.yaml) 文件复制到当前目录下, 然后执行:

```bash
accelerate launch --config_file default_config.yaml ddp_accelerate_advanced.py
```

**效果图如下:**
![deepspeed result](/assets/ddp_deepspeed_result.png)


<br>
<br>


### 使用 DeepSpeed与 没有使用 DeepSpeed 的 accelerate 对比

没有使用 `DeepSpeed`:

![no deepspeed](/assets/no_deepspeed_result.png)



使用 `DeepSpeed`:
![use deepspeed](/assets/deepspeed_result.png)


可以看到时间还增加了几秒，可能是多了通信时间吧，在 `epoch` 大的时候才能拉开差距


<br>
<br>


### DeepSpeed 与 Accelerate 使用的一些细节

> **Tips:**
> `.yaml` 是 `accelerate` 使用的配置文件， `.json` 才是 `DeepSpeed` 使用的配置文件

<br>
<br>

**使用 stage2：**

之前使用 `accelerate config` 进行配置的时候，并没有指定 `DeepSpeed` 的 `config` 文件，我们可以从 `accelerate` [官方文档](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) 中找到如何配置


在 [`default_config.yaml`](/distributed_train/DeepSpeed/default_config.yaml) 基础上，修改为 [`default_config_modify_zero2.yaml`](/distributed_train/DeepSpeed/default_config_modify_zero2.yaml), 并根据官方案例 编写 [`zero_stage2_config.json`](/distributed_train/DeepSpeed/zero_stage2_config.json), 删去了 `optimizer` 和 `scheduler` 部分, 并且将 `fp16` 修改为 `bf16`, 删除其余参数，只保留 `"enabled": true`


> 官网的 `config` 文件案例有点问题，指定了 `deepspeed_config_file` 需要注释掉 `mixed_precision: bf16`，否则会报错


<br>
<br>

**使用 stage3：**

- 方式1:

**不使用 `deepspeed_config_file`, 直接修改 `zero_stage: 3`.**

如果按照之前的配置执行代码会报梯度相关的错误，原因是 `stage3` 是将模型拆分进行训练的，需要在 `deepspeed_config:` 中增加 `zero3_save_16bit_model: true`, 详细请看 [`default_config_zero3.yaml`](/distributed_train/DeepSpeed/default_config_zero3.yaml)


还有一个问题就是在 执行代码 `ddp_acclerate_advanced.py` 结束一个 `epoch` 后报错，原因是在 `ddp_acclerate_advanced.py` 中使用了 `torch.inference_mode` 完全去掉了梯度，只要改为 `torch.no_grad` 即可.

错误如下所示：
![no grad error](/assets/no_grad_error.png)

> 这个错误引出了 `torch.inference_mode` 与 `torch.no_grad` 的区别


<br>

- 方式2:

**用 `deepspeed_config_file` 方式 启动 `stage3`.**


修改 `deepspeed_config_file` 为 `zero_stage3_config.json`, 在 `zero_optimization`部分 修改参数为 `"stage": 3`, 并且增加参数 `"stage3_gather_16bit_weights_on_mode_save": true`


<br>
<br>


**更细粒度的使用方式：**

`zero stage2`, 执行 :
```bash
accelerate launch --config_file default_config.yaml ddp_accelerate_advanced.py
```

`zero stage2` 指定 deepspeed config file 形式, 执行 :
```bash
accelerate launch --config_file default_config_modify_zero2.yaml ddp_accelerate_advanced.py
```


`zero stage3` 以此类推.

> **Tips:** `zero2` 我命名为 `default_config.yaml`, `zero3` 命名为 `default_config_zero3.yaml`


<br>
<br>


**额外地:**

如果直接使用 `tranformers` 封装好的代码 [`ddp_transformers.py`](/distributed_train/distributed_data_parallel/ddp_transformers.py), 同样的直接执行:

```bash
accelerate launch --config_file default_config.yaml ddp_transformers.py
```

训练过程如下图所示:

![trainer in deepspeed](/assets/deepspeed_trainer_result.png)

感觉还是 `transformers` 一套集成起来的好看～


config文件根据自己的需要进行修改即可, 需要注意的是 `ddp_transformers` 中的 `TrainingArguments` 参数里面配置的值必须与 `default_config.yaml` 文件的信息一直，否则会报错～



<br>
<br>

### Accelerate && DeepSpeed 多机多卡使用

两种情况:

- 情景1:

直连情况下，指定 `deepspeed_hostfile`, `num_machines`, `num_process`, `main_process_port`, 正常通过 `Accelerate` 启动

<br>

- 情景2:

slurm 管理情况下, 配置启动器为 `torchrun` 标准的启动器, 指定 `num_machines`, `machine_rank`, `num_process`, `main_process_port`, `main_process_ip`, 还是以 `Accelerate` 启动, 模式与 `torchrun` 启动类似