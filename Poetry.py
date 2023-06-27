#!/usr/bin/env python
# coding: utf-8

# ## MindSpore-BERT-Text Generation
# ### 1. 下载源码和数据至本地容器
# 
# 因为notebook是挂载在obs上，运行的容器实例不能直接读取操作obs上的文件，需下载至容器本地环境中

# In[1]:


import moxing as mox
mox.file.copy_parallel(src_url="s3://ascend-zyjs-dcyang/nlp/text_generation_mindspore/data/", dst_url='./data/') 
mox.file.copy_parallel(src_url="s3://ascend-zyjs-dcyang/nlp/text_generation_mindspore/src/", dst_url='./src/') 


# ### 2. 导入依赖库

# In[3]:


import os
import re
import time
import numpy as np

import mindspore.dataset as de
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C

from mindspore import context
from mindspore import log as logger
from mindspore.train.model import Model
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export
from mindspore.common.parameter import Parameter
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from easydict import EasyDict as edict
from src.bert_model import BertConfig
from src.poetry_dataset import create_tokenizer, padding
from src.utils import BertPoetry, BertPoetryCell, BertLearningRate, BertPoetryModel, LossCallBack


# ### 3. 设置运行环境

# In[4]:


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)


# ### 4. 定义配置参数

# In[5]:


bs = 16

cfg = edict({
    'dict_path': './data/vocab.txt',
    'disallowed_words': ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']'],
    'max_len': 64,
    'min_word_frequency': 8,
    'dataset_path': './data/poetry.txt',
    'batch_size': bs,
    'epoch_num': 2,
    'ckpt_prefix': 'poetry',
    'ckpt_dir': './data/checkpoint/',
    'pre_training_ckpt': './data/pretrain_ckpt/bert_base.ckpt',
    'optimizer': 'AdamWeightDecayDynamicLR',
    'AdamWeightDecay': edict({
        'learning_rate': 3e-5,
        'end_learning_rate': 1e-10,
        'power': 1.0,
        'weight_decay': 1e-5,
        'eps': 1e-6,
    }),
    'Lamb': edict({
        'start_learning_rate': 2e-5,
        'end_learning_rate': 1e-7,
        'power': 1.0,
        'weight_decay': 0.01,
        'decay_filter': lambda x: False,
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }),
})

bert_net_cfg = BertConfig(
    batch_size=bs,
    seq_length=128,
    vocab_size=3191,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    input_mask_from_dataset=True,
    token_type_ids_from_dataset=True,
    dtype=mstype.float32,
    compute_type=mstype.float16,
)


# ### 5. 加载数据
# 
# 定义数据加载函数

# In[6]:


class PoetryDataGenerator(object):
    def __init__(self, batch_size, poetry, tokenizer, length=128):
        self.data = poetry
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.length = length

    def __getitem__(self, index):
        np.random.shuffle(self.data)
        current_data = self.data[index]

        token_ids, segment_ids = self.tokenizer.encode(current_data)
        batch_token_ids = padding(token_ids, length=self.length)
        batch_segment_ids = padding(segment_ids, length=self.length)
        pad_mask = (batch_token_ids != 0).astype(np.float32)
        return (batch_token_ids, batch_segment_ids, pad_mask)

    def __len__(self):
        return len(self.data)


def create_poetry_dataset(batch_size, poetry, tokenizer):
    dt = PoetryDataGenerator(batch_size, poetry, tokenizer)
    ds = de.GeneratorDataset(dt, ["input_ids", "token_type_id", "pad_mask"])
    #ds.set_dataset_size(dt.__len__())
    int_type_cast_op = C.TypeCast(mstype.int32)
    float_type_cast_op = C.TypeCast(mstype.float32)
    ds = ds.map(input_columns="input_ids", operations=int_type_cast_op)
    ds = ds.map(input_columns="token_type_id", operations=int_type_cast_op)
    ds = ds.map(input_columns="pad_mask", operations=float_type_cast_op)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


# 加载数据

# In[7]:


poetry, tokenizer, keep_words = create_tokenizer(cfg)
num_tokens = len(keep_words)
print(num_tokens)

dataset = create_poetry_dataset(bert_net_cfg.batch_size, poetry, tokenizer)


# 查看数据样本

# In[8]:


next(dataset.create_dict_iterator())


# ### 6. 模型训练
# 
# 定义训练函数：

# In[9]:


def train():
    '''
    build bert model for poetry generation
    '''

    poetrymodel = BertPoetryModel(bert_net_cfg, True, num_tokens, dropout_prob=0.1)
    netwithloss = BertPoetry(poetrymodel, bert_net_cfg, True, dropout_prob=0.1)
    
    callback = LossCallBack(poetrymodel)

    # optimizer
    steps_per_epoch = dataset.get_dataset_size()
    print("============ steps_per_epoch is {}".format(steps_per_epoch))
    lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=1000,
                                       decay_steps=cfg.epoch_num*steps_per_epoch,
                                       power=cfg.AdamWeightDecay.power)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr_schedule)
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.ckpt_prefix, directory=cfg.ckpt_dir, config=ckpt_config)
    param_dict = load_checkpoint(cfg.pre_training_ckpt)
    new_dict = {}

    # load corresponding rows of embedding_lookup
    for key in param_dict:
        if "bert_embedding_lookup" not in key:
            new_dict[key] = param_dict[key]
        else:
            value = param_dict[key]
            np_value = value.data.asnumpy()
            np_value = np_value[keep_words]
            tensor_value = Tensor(np_value, mstype.float32)
            parameter_value = Parameter(tensor_value, name=key)
            new_dict[key] = parameter_value

    load_param_into_net(netwithloss, new_dict)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertPoetryCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    
    model.train(cfg.epoch_num, dataset, callbacks=[callback, ckpoint_cb], dataset_sink_mode=True)


# 启动训练：

# In[10]:


train()


# ### 7. 测试评估
# 
# 定义藏头诗生成函数

# In[11]:


def generate_head_poetry(model, head=""):
    token_ids, segment_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]
    segment_ids = segment_ids[:-1]
    punctuations = ['，', '。']
    punctuation_ids = [tokenizer._token_to_id[token] for token in punctuations]
    poetry = []
    length = 128

    for ch in head:
        poetry.append(ch)
        token_id = tokenizer._token_to_id[ch]
        token_ids.append(token_id)
        segment_ids.append(0)
        while True:
            index = len(token_ids)
            _target_ids = padding(np.array(token_ids), length=length)
            _segment_ids = padding(np.array(segment_ids), length=length)
            pad_mask = (_target_ids != 0).astype(np.float32)
            
            _target_ids = Tensor([_target_ids], mstype.int32)
            _segment_ids = Tensor([_segment_ids], mstype.int32)
            pad_mask = Tensor([pad_mask], mstype.float32)
            _probas = model(_target_ids, _segment_ids, pad_mask).asnumpy()
            
            _probas = _probas[0, index-1, 3:]
            p_args = _probas.argsort()[::-1][:100]
            p = _probas[p_args]
            p = p / sum(p)
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index] + 3
            token_ids.append(target)
            segment_ids.append(0)
            if target > 3:
                poetry.append(tokenizer._id_to_token[target])
            if target in punctuation_ids:
                break
    return ''.join(poetry)


# 查看模型checkpoint保存路径

# In[12]:


get_ipython().system('ls data/checkpoint')


# 加载恢复离线模型

# In[13]:


bert_net_cfg.batch_size = 1
poetrymodel = BertPoetryModel(bert_net_cfg, False, 3191, dropout_prob=0.0)
poetrymodel.set_train(False)
ckpt_path = './data/checkpoint/poetry-2_1535.ckpt'
param_dict = load_checkpoint(ckpt_path)
load_param_into_net(poetrymodel, param_dict)


# 测试：

# In[14]:


generate_head_poetry(poetrymodel, "人工智能")


# In[15]:


generate_head_poetry(poetrymodel, "自然语言处理")


# In[ ]:




