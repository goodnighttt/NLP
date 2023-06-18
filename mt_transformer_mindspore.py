#!/usr/bin/env python
# coding: utf-8

# ## MindSpore-Transformer-Machine Translation
# ### 1. 下载源码和数据至本地容器
# 
# 因为notebook是挂载在obs上，运行的容器实例不能直接读取操作obs上的文件，需下载至容器本地环境中

# In[1]:


import moxing as mox
mox.file.copy_parallel(src_url="obs://nlp-workspace/2.nmt/transformer/data/", dst_url='./data/')
mox.file.copy_parallel(src_url="obs://nlp-workspace/2.nmt/transformer/src/", dst_url='./src/')


# ### 2. 导入依赖库

# In[2]:


import os
import numpy as np
from easydict import EasyDict as edict

import mindspore.nn as nn
from mindspore import context
import mindspore.dataset.engine as de
import mindspore.common.dtype as mstype
from mindspore.mindrecord import FileWriter
from mindspore.common.parameter import Parameter
import mindspore.dataset.transforms.c_transforms as deC
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import tokenization
from src.train_util import LossCallBack
from src.lr_schedule import create_dynamic_lr
from src.transformer_model import TransformerConfig, TransformerModel
from src.data_utils import create_training_instance, write_instance_to_file
from src.transformer_for_train import TransformerTrainOneStepCell, TransformerNetworkWithLoss, TransformerTrainOneStepWithLossScaleCell


# ### 3. 设置运行环境

# In[3]:


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


# ### 4. 定义数据处理相关参数

# In[4]:


data_cfg = edict({
        'input_file': './data/ch_en_all.txt',
        'vocab_file': './data/ch_en_vocab.txt',
        'train_file_mindrecord': './data/train.mindrecord',
        'eval_file_mindrecord': './data/test.mindrecord',
        'train_file_source': './data/source_train.txt',
        'eval_file_source': './data/source_test.txt',
        'num_splits':1,
        'clip_to_max_len': False,
        'max_seq_length': 40
})


# ### 5. 定义数据处理函数
# 
# 加载原始数据，切分训练、测试数据，并预处理成模型输入所需的数据形式，并保存为mindrecord格式

# In[5]:


def data_prepare(cfg, eval_idx):
    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=cfg.vocab_file)

    writer_train = FileWriter(cfg.train_file_mindrecord, cfg.num_splits)
    writer_eval = FileWriter(cfg.eval_file_mindrecord, cfg.num_splits)
    data_schema = {"source_sos_ids": {"type": "int32", "shape": [-1]},
                   "source_sos_mask": {"type": "int32", "shape": [-1]},
                   "source_eos_ids": {"type": "int32", "shape": [-1]},
                   "source_eos_mask": {"type": "int32", "shape": [-1]},
                   "target_sos_ids": {"type": "int32", "shape": [-1]},
                   "target_sos_mask": {"type": "int32", "shape": [-1]},
                   "target_eos_ids": {"type": "int32", "shape": [-1]},
                   "target_eos_mask": {"type": "int32", "shape": [-1]}
                   }

    writer_train.add_schema(data_schema, "tranformer train")
    writer_eval.add_schema(data_schema, "tranformer eval")

    index = 0
    f_train = open(cfg.train_file_source, 'w', encoding='utf-8')
    f_test = open(cfg.eval_file_source,'w',encoding='utf-8')
    f = open(cfg.input_file, "r", encoding='utf-8')
    for s_line in f:
        print("finish {}/{}".format(index, 23607), end='\r')
        
        line = tokenization.convert_to_unicode(s_line)

        source_line, target_line = line.strip().split("\t")
        source_tokens = tokenizer.tokenize(source_line)
        target_tokens = tokenizer.tokenize(target_line)

        if len(source_tokens) >= (cfg.max_seq_length-1) or len(target_tokens) >= (cfg.max_seq_length-1):
            if cfg.clip_to_max_len:
                source_tokens = source_tokens[:cfg.max_seq_length-1]
                target_tokens = target_tokens[:cfg.max_seq_length-1]
            else:
                continue
        
        index = index + 1
        # print(source_tokens)
        instance = create_training_instance(source_tokens, target_tokens, cfg.max_seq_length)
        
        if index in eval_idx:
            f_test.write(s_line)
            features = write_instance_to_file(writer_eval, instance, tokenizer, cfg.max_seq_length)
        else:
            f_train.write(s_line)
            features = write_instance_to_file(writer_train, instance, tokenizer, cfg.max_seq_length)
    f.close()
    f_test.close()
    f_train.close()
    writer_train.commit()
    writer_eval.commit()


# ### 6. 数据处理，随机选20%作为测试数据

# In[6]:


sample_num = 23607
eval_idx = np.random.choice(sample_num, int(sample_num*0.2), replace=False)
data_prepare(data_cfg, eval_idx)


# ### 7. 定义数据加载函数

# In[7]:


def load_dataset(batch_size=1, data_file=None):
    """
    Load mindrecord dataset
    """
    ds = de.MindDataset(data_file,
                        columns_list=["source_eos_ids", "source_eos_mask",
                                      "target_sos_ids", "target_sos_mask",
                                      "target_eos_ids", "target_eos_mask"],
                        shuffle=False)
    type_cast_op = deC.TypeCast(mstype.int32)
    ds = ds.map(input_columns="source_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="source_eos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_mask", operations=type_cast_op)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds.channel_name = 'transformer'
    return ds


# 测试数据是否能正常加载

# In[8]:


next(load_dataset(data_file=data_cfg.train_file_mindrecord).create_dict_iterator())['source_eos_ids'][0]


# ### 8. 定义训练相关配置参数

# In[9]:


train_cfg = edict({
    #--------------------------------------nework confige-------------------------------------
    'transformer_network': 'base',
    'init_loss_scale_value': 1024,
    'scale_factor': 2,
    'scale_window': 2000,

    'lr_schedule': edict({
        'learning_rate': 1.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),
    #-----------------------------------save model confige-------------------------
    'enable_save_ckpt': True ,        #Enable save checkpointdefault is true.
    'save_checkpoint_steps':590,   #Save checkpoint steps, default is 590.
    'save_checkpoint_num':2,     #Save checkpoint numbers, default is 2.
    'save_checkpoint_path': './checkpoint',    #Save checkpoint file path,default is ./checkpoint/
    'save_checkpoint_name':'transformer-32_40',
    'checkpoint_path':'',     #Checkpoint file path
    
    
    #-------------------------------device confige-----------------------------
    'enable_data_sink':False,   #Enable data sink, default is False.
    'device_id':0,
    'device_num':1,
    'distribute':False,
    
    # -----------------mast same with the dataset-----------------------
    'seq_length':40,
    'vocab_size':10067,
    
    #--------------------------------------------------------------------------
    'data_path':"./data/train.mindrecord",   #Data path
    'epoch_size':15,
    'batch_size':32,
    'max_position_embeddings':40,
    'enable_lossscale': False,       #Use lossscale or not, default is False.
    'do_shuffle':True       #Enable shuffle for dataset, default is True.
})
'''
two kinds of transformer model version
'''
if train_cfg.transformer_network == 'base':
    transformer_net_cfg = TransformerConfig(
        batch_size=train_cfg.batch_size,
        seq_length=train_cfg.seq_length,
        vocab_size=train_cfg.vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="relu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=train_cfg.max_position_embeddings,
        initializer_range=0.02,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
        dtype=mstype.float32,
        compute_type=mstype.float16)
elif train_cfg.transformer_network == 'large':
    transformer_net_cfg = TransformerConfig(
        batch_size=train_cfg.batch_size,
        seq_length=train_cfg.seq_length,
        vocab_size=train_cfg.vocab_size,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="relu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=train_cfg.max_position_embeddings,
        initializer_range=0.02,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
        dtype=mstype.float32,
        compute_type=mstype.float16)
else:
    raise Exception("The src/train_confige of transformer_network must base or large. Change the str/train_confige file and try again!")


# ### 9. 定义训练函数

# In[10]:


def train(cfg):
    """
    Transformer training.
    """
    
    train_dataset = load_dataset(cfg.batch_size, data_file=cfg.data_path)

    netwithloss = TransformerNetworkWithLoss(transformer_net_cfg, True)

    if cfg.checkpoint_path:
        parameter_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(netwithloss, parameter_dict)

    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=train_dataset.get_dataset_size()*cfg.epoch_size,
                                  learning_rate=cfg.lr_schedule.learning_rate,
                                  warmup_steps=cfg.lr_schedule.warmup_steps,
                                  hidden_size=transformer_net_cfg.hidden_size,
                                  start_decay_step=cfg.lr_schedule.start_decay_step,
                                  min_lr=cfg.lr_schedule.min_lr), mstype.float32)
    optimizer = Adam(netwithloss.trainable_params(), lr)

    callbacks = [TimeMonitor(train_dataset.get_dataset_size()), LossCallBack()]
    if cfg.enable_save_ckpt:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                       keep_checkpoint_max=cfg.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix=cfg.save_checkpoint_name, directory=cfg.save_checkpoint_path, config=ckpt_config)
        callbacks.append(ckpoint_cb)

    if cfg.enable_lossscale:
        scale_manager = DynamicLossScaleManager(init_loss_scale=cfg.init_loss_scale_value,
                                                scale_factor=cfg.scale_factor,
                                                scale_window=cfg.scale_window)
        update_cell = scale_manager.get_update_cell()
        netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,scale_update_cell=update_cell)
    else:
        netwithgrads = TransformerTrainOneStepCell(netwithloss, optimizer=optimizer)

    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    model.train(cfg.epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=cfg.enable_data_sink)


# ### 10. 启动训练

# In[16]:


train(train_cfg)


# ### 11. 定义推理相关参数配置

# In[23]:


eval_cfg = edict({
    'transformer_network': 'base',
    
    'data_file': './data/test.mindrecord',
    'test_source_file':'./data/source_test.txt',
    'model_file': './checkpoint/transformer-32_40-15_590.ckpt' ,
    'vocab_file':'./data/ch_en_vocab.txt',
    'token_file': './token-32-40.txt',
    'pred_file':'./pred-32-40.txt',
    
    # -------------------mast same with the train config and the datsset------------------------
    'seq_length':40,
    'vocab_size':10067,

    #-------------------------------------eval config-----------------------------
    'batch_size':32,
    'max_position_embeddings':40       # mast same with the train config
})

'''
two kinds of transformer model version
'''
if eval_cfg.transformer_network == 'base':
    transformer_net_cfg = TransformerConfig(
        batch_size=eval_cfg.batch_size,
        seq_length=eval_cfg.seq_length,
        vocab_size=eval_cfg.vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=eval_cfg.max_position_embeddings,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
        beam_width=4,
        max_decode_length=eval_cfg.seq_length,
        length_penalty_weight=1.0,
        dtype=mstype.float32,
        compute_type=mstype.float16)
    
elif eval_cfg.transformer_network == 'large':
    transformer_net_cfg = TransformerConfig(
        batch_size=eval_cfg.batch_size,
        seq_length=eval_cfg.seq_length,
        vocab_size=eval_cfg.vocab_size,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=eval_cfg.max_position_embeddings,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
        beam_width=4,
        max_decode_length=80,
        length_penalty_weight=1.0,
        dtype=mstype.float32,
        compute_type=mstype.float16)
else:
    raise Exception("The src/eval_confige of transformer_network must base or large and same with the train_confige confige. Change the str/eval_confige file and try again!")


# ### 12. 定义评估测试函数

# In[24]:


class TransformerInferCell(nn.Cell):
    """
    Encapsulation class of transformer network infer.
    """
    def __init__(self, network):
        super(TransformerInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self,
                  source_ids,
                  source_mask):
        predicted_ids = self.network(source_ids, source_mask)
        return predicted_ids

def load_weights(model_path):
    """
    Load checkpoint as parameter dict, support both npz file and mindspore checkpoint file.
    """
    if model_path.endswith(".npz"):
        ms_ckpt = np.load(model_path)
        is_npz = True
    else:
        ms_ckpt = load_checkpoint(model_path)
        is_npz = False

    weights = {}
    for msname in ms_ckpt:
        infer_name = msname
        if "tfm_decoder" in msname:
            infer_name = "tfm_decoder.decoder." + infer_name
        if is_npz:
            weights[infer_name] = ms_ckpt[msname]
        else:
            weights[infer_name] = ms_ckpt[msname].data.asnumpy()
    weights["tfm_decoder.decoder.tfm_embedding_lookup.embedding_table"] =         weights["tfm_embedding_lookup.embedding_table"]

    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name]), name=name)
    return parameter_dict

def evaluate(cfg):
    """
    Transformer evaluation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", reserve_class_name_in_scope=False)

    tfm_model = TransformerModel(config=transformer_net_cfg, is_training=False, use_one_hot_embeddings=False)
    print(cfg.model_file)
    parameter_dict = load_weights(cfg.model_file)
    load_param_into_net(tfm_model, parameter_dict)
    tfm_infer = TransformerInferCell(tfm_model)
    model = Model(tfm_infer)
    
    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=cfg.vocab_file)
    dataset = load_dataset(batch_size=cfg.batch_size, data_file=cfg.data_file) 
    predictions = []
    source_sents = []
    target_sents = []
    f2 = open(cfg.test_source_file, 'r', encoding='utf-8')
    for batch in dataset.create_dict_iterator():
        source_sents.append(batch["source_eos_ids"])
        target_sents.append(batch["target_eos_ids"])
        source_ids = Tensor(batch["source_eos_ids"], mstype.int32)
        source_mask = Tensor(batch["source_eos_mask"], mstype.int32)
        predicted_ids = model.predict(source_ids, source_mask)
        #predictions.append(predicted_ids.asnumpy())
        # ----------------------------------------decode and write to file(token file)---------------------
        batch_out = predicted_ids.asnumpy()
        for i in range(transformer_net_cfg.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]
            token_ids = [str(x) for x in batch_out[i].tolist()]
            token=" ".join(token_ids)
            #-------------------------------token_ids to real output file-------------------------------
            token_ids = [int(x) for x in token.strip().split()]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            sent = " ".join(tokens)
            sent = sent.split("<s>")[-1]
            sent = sent.split("</s>")[0]
            
            label_sent = f2.readline().strip()+'\t'
            print("source: {}".format(label_sent))
            print("result: {}".format(sent.strip()))


# ### 13. 启动评估测试

# In[25]:


evaluate(eval_cfg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




