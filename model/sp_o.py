import codecs
from os.path import dirname, abspath
import sys
import numpy as np
import torch
import torch.nn as nn
from numpy.random import randint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')
from utils.function import *
from utils.logger import Logger
from embedding.torchEmbedding import TorchEmbedding, PositionEmbedding, EmbeddingProcessor
from model.attention import SelfAttention, MultiHeadAttention
from model.loss_function import DynamicFocalLossBCE, FocalLossBCE
from model.Decoder import DGCNNDecoder
from model.Encoder import RNNEncoder
from model.model_utils import EMA
from config import Config
from model.evaluation import evaluate

parent_path = abspath(dirname(dirname(__file__)))

dataset = Config.dataset

TRAIN_FILE = parent_path + '/data/' + dataset + '/my_train_data.json'
# DEV_FILE = parent_path + '/data/' + dataset + '/my_dev_data.json'
TEST_FILE = parent_path + '/data/' + dataset + '/my_test_data.json'
ID_PREDICATE_FILE = parent_path + '/data/' + dataset + '/id_and_predicate_no_unk.json'
PREDICATE_INFO_FILE = parent_path + '/data/' + dataset + '/predicate_info.json'
CHAR2ID_FILE = parent_path + '/data/' + dataset + '/char2id.json'

log = Logger('sp_o_{}'.format(dataset), std_out=False, save2file=True).get_logger()


class TextData(object):
    '''
    加载文本文件到内存，每条数据对应一个输入、一个输出
    json_file：json文件的绝对路径
    id_predicate_file: id_predicate_file
    '''

    def __init__(self, json_file: str, id_predicate_file: str = ID_PREDICATE_FILE, char2id_file: str = CHAR2ID_FILE):
        '''
        json_file：json文件的绝对路径
        '''
        # python子类的构造方法一定会调用父类的构造方法
        super().__init__()
        raw_data = read_json(json_file)

        self.predicate_info = read_json(PREDICATE_INFO_FILE)
        self.id2predicate, self.predicate2id = read_json(id_predicate_file)
        with codecs.open(char2id_file, 'r', encoding='utf-8') as f:
            self.char2id = ujson.load(f)

        # 关系总数
        # 0是unk类
        self.num_predicate = len(self.id2predicate)
        self.max_seq_len, self.subject_max_len = self.__compute_max_len(raw_data)
        self.len = len(raw_data)

        # 提取出(text, spo_list, choice_index)，这三个都是list
        self.inputs_outputs = self.__process_data_for_dataloader(raw_data)

    # 处理成 输入到 dataloader 的形式
    def __process_data_for_dataloader(self, raw_data: list):
        '''
        '''
        text = []
        spo_list = []
        choice_index = []

        print('process data ...')
        for data in raw_data:
            text.append(data['text'])
            spo_list.append(data['spo_list'])

            # 随机采样一个sp作为o模型的输入
            choice_index.append(randint(low=0, high=len(data['spo_list'])))

        return (text, spo_list, choice_index)

    def __compute_max_len(self, raw_data: list):
        """
        计算所有样本最大句子长度 和 最大主实体长度
        """
        max_seq_len = 0
        subject_max_len = 0
        for data in raw_data:
            max_seq_len = max(len(data['text'].strip().split(' ')), max_seq_len)
            # 取了一个样本data 再从spolist中取一个spo字典，再从字典中取一个subject
            subject_max_len = max(max([len(spo['subject'].strip().split(' ')) for spo in data['spo_list']]),
                                  subject_max_len)
        return max_seq_len, subject_max_len

    def get_inupts_outputs(self):
        return self.inputs_outputs

    def __len__(self):
        return self.len


class SpoDataset(Dataset):
    """
    预处理
    """

    def __init__(self, inputs_outputs: list, predicate_info: dict, predicate2id: dict, max_seq_len: int):
        '''
        '''
        super(SpoDataset, self).__init__()
        text, spo_list, choice_index = inputs_outputs

        self.text = text
        self.spo_list = spo_list
        self.choice_index = choice_index

        self.predicate_info = predicate_info
        self.max_seq_len = max_seq_len
        self.num_predicate = len(predicate2id)
        self.predicate2id = predicate2id

        self.len = len(text)
        pass

    def __getitem__(self, index):
        """
        重写Dataset的方法__getitem__()
        index为样本的索引，item可以理解为样本
        getitem：返回index对应样本的某些参数

        目前类成员是 各种数据的 所有的样本的list，__getitem__ 提供了index
        利用这些list和index，返回一个样本的各种数据
        各种数据：text, 输入到o的text, sp_start, sp_end, o_start, o_end, len(text)
        """
        text = self.text[index]
        spo_list = self.spo_list[index]

        # 一个sp作为o模型的输入
        # 固定采样
        # choice_index = self.choice_index[index]

        # 训练时随机采样
        choice_index = randint(low=0, high=len(spo_list))

        max_len = self.max_seq_len
        preidcate2id = self.predicate2id
        spo_choose = spo_list[choice_index]
        s_choose = spo_choose['subject']
        p_choose = spo_choose['predicate']

        # 初始化sp_start sp_end o_start o_end为0
        # sp可以理解为二维数组，（200,49）  200个位置，49个关系    来表示一个样本的s和p
        # 一个text 对应一个sp_start sp_end o_start o_end
        sp_start = [[0.0] * self.num_predicate for _ in range(max_len)]
        sp_end = [[0.0] * self.num_predicate for _ in range(max_len)]
        o_start = [0.0] * max_len
        o_end = [0.0] * max_len

        # 将数据集中的位置信息编码进sp数组中
        for spo in spo_list:
            predicate_id = preidcate2id[spo['predicate']]
            sp_start[spo['subject_start']][predicate_id] = 1.0
            sp_end[spo['subject_end'] - 1][predicate_id] = 1.0
            # 挑出被选中的spo，但这里只选中了 sp和选中spo相同的spo，也就是sp可能有多个不同的o
            # 只将与 spo_choose 具有相同sp 的spo 的o 赋值
            # 一个样本中只有一个 spo_choose
            if spo['subject'] == s_choose and spo['predicate'] == p_choose:
                o_start[spo['object_start']] = 1.0
                o_end[spo['object_end'] - 1] = 1.0

        p_info = self.predicate_info[p_choose]
        # s种类，s，p，o种类，原文本
        o_query_text = '{}，{}，{}，{}。{}'.format(p_info['s_type'], s_choose, p_choose, p_info['o_type'], text)

        return text, o_query_text, sp_start, sp_end, o_start, o_end, len(text.strip().split(' '))

    def __len__(self):
        return self.len


def collate_fn(data):
    '''
    data为一个二维数组，一个batch的(__getitem__返回值)
    将各个维度 整理成 各个list，将这些list组合成dict
    '''

    lens = [item[6] for item in data]
    max_len = max(lens)

    text = [item[0] for item in data]
    o_query_text = [item[1] for item in data]

    sp_start = [item[2][0: max_len] for item in data]
    sp_end = [item[3][0: max_len] for item in data]

    o_start = [item[4][0: max_len] for item in data]
    o_end = [item[5][0: max_len] for item in data]

    as_tensor = torch.as_tensor

    ret = {
        'text': text,
        'o_query_text': o_query_text,
        'sp_start': as_tensor(sp_start),
        'sp_end': as_tensor(sp_end),
        'o_start': as_tensor(o_start),
        'o_end': as_tensor(o_end),
    }

    return ret


class SubjectPredicateModel(nn.Module):
    """
    预测主实体和关系的模型sp
    """

    def __init__(self, embedding_size: int, num_predicate: int, num_heads: int, rnn_type: str, rnn_hidden_size: int,
                 forward_dim: int, device: str = 'cuda', dropout_prob: float = 0.1):
        '''
        embedding_size： 词向量的大小
        num_predicate: 模型要预测的关系总数
        '''
        super(SubjectPredicateModel, self).__init__()

        self.embedding_encoder = RNNEncoder(
            embedding_dim=embedding_size,
            num_layers=2,
            rnn_type=rnn_type,
            hidden_size=rnn_hidden_size,
            dropout_prob=dropout_prob,
        )

        self.attention = SelfAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            device=device,
        )

        d = [1, 2, 3, 1, 2, 3]
        k = [5, 5, 5, 3, 3, 3]
        self.cnn = DGCNNDecoder(
            dilations=d,
            kernel_sizes=k,
            in_channels=embedding_size,
        )

        forward_dim_in = embedding_size * 1
        self.sp_start_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, num_predicate),
        )

        self.sp_end_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, num_predicate),
        )

    def forward(self, input_embedding: Tensor, mask: Tensor, position_embedding: Tensor):
        '''
        '''

        share_feature = self.embedding_encoder(
            input_embedding=input_embedding,
            position_embedding=position_embedding,
            mask=mask,
        )

        # ===========================================================================================#

        outs, _ = self.attention(
            inputs=share_feature,
            mask=mask,
        )
        # Ablation study: without cnn
        # outs = self.cnn(outs)

        sp_start = self.sp_start_fc(outs)
        sp_end = self.sp_end_fc(outs)

        return share_feature, sp_start, sp_end


class ObjectModel(nn.Module):
    """
    预测客体的模型o
    """

    def __init__(self, embedding_size: int, num_heads: int, rnn_type: str, rnn_hidden_size: int, forward_dim: int = 128,
                 device: str = 'cuda'):
        '''
        '''
        super(ObjectModel, self).__init__()

        self.embedding_encoder = RNNEncoder(
            embedding_dim=embedding_size,
            num_layers=2,
            rnn_type=rnn_type,
            hidden_size=rnn_hidden_size,
        )

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            device=device,
        )

        d = [1, 2, 3, 1, 2, 3]
        k = [5, 5, 5, 3, 3, 3]
        self.cnn = DGCNNDecoder(
            dilations=d,
            kernel_sizes=k,
            in_channels=embedding_size,
        )

        forward_dim_in = embedding_size * 1
        self.object_start_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, 1),
        )

        self.object_end_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, 1),
        )

    def forward(self, share_feature: Tensor, share_mask: Tensor, query_embedding: Tensor, query_mask: Tensor,
                query_pos_embedding: Tensor):
        '''
        '''

        rnn_out = self.embedding_encoder(
            input_embedding=query_embedding,
            position_embedding=query_pos_embedding,
            mask=query_mask,
        )

        # (Q,K,V)  Q:share_feature K:rnn_out V:rnn_out
        # 多头注意力融合 SP模型输出share_feature 和 o模型输入o_query
        outs, _ = self.multi_head_attention(
            query=share_feature,
            key=rnn_out,
            value=rnn_out,
            mask=query_mask,
        )

        # Ablation study: without cnn
        # outs = self.cnn(outs)

        object_start = self.object_start_fc(outs)
        object_end = self.object_end_fc(outs)

        return object_start, object_end


# ==================================================== end model ========================================================#


class Trainer(object):
    """
    训练器，调用sp模型和o模型
    """

    def __init__(self):
        '''
        sp_o模型训练
        '''
        super().__init__()
        self.train_data = TextData(TRAIN_FILE, ID_PREDICATE_FILE)
        # self.dev_data = read_json(DEV_FILE)  # 评估数据直接用原始数据
        self.test_data = read_json(TEST_FILE)

        # 统一使用训练集的长度
        # TextData中的成员 传到 Trainer中
        self.max_seq_len = self.train_data.max_seq_len
        self.num_predicate = self.train_data.num_predicate

        self.id2predicate = self.train_data.id2predicate
        self.predicate2id = self.train_data.predicate2id
        self.predicate_info = self.train_data.predicate_info

    def train(self, config: Config, device):

        # 训练数据加载器
        train_data_loader = DataLoader(
            dataset=SpoDataset(
                inputs_outputs=self.train_data.inputs_outputs,
                predicate_info=self.predicate_info,
                predicate2id=self.predicate2id,
                max_seq_len=self.max_seq_len,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,  # win10下，DataLoader使用lmdb进行多进程加载词向量会报错
            collate_fn=collate_fn,  # 不自己写collate_fn处理numpy到tensor的转换会导致数据转换非常慢
            pin_memory=True,
        )

        # 预训练的路径
        base_path = parent_path + '/model_file'

        # torch_embedding
        # 创建一个embedding类（继承Module），类成员self.embedding也是一个Module
        if config.from_pretrained == 'none':
            embedding = TorchEmbedding(config.embedding_size, device).to(device)
        elif config.from_pretrained == 'NYT':
            config.embedding_size = 100
            config.num_heads = 10
            config.rnn_hidden_size = 200
            pretrain_file = base_path + '/pretrain/NYT100.npy'
            embedding = TorchEmbedding(config.embedding_size, device, char2id_file=CHAR2ID_FILE,
                                       from_pretrain=pretrain_file).to(device)
        elif config.from_pretrained == 'WebNLG':
            config.embedding_size = 100
            config.num_heads = 10
            config.rnn_hidden_size = 200
            pretrain_file = base_path + '/pretrain/WebNLG100.npy'
            embedding = TorchEmbedding(config.embedding_size, device, char2id_file=CHAR2ID_FILE,
                                       from_pretrain=pretrain_file).to(device)
        else:
            raise ValueError('value config.from_pretrained "{}" error.'.format(config.from_pretrained))

        if config.embedding_size % config.num_heads != 0:
            raise ValueError(('隐藏层的维度({})不是注意力头个数({})的整数倍'.format(config.embedding_size, config.num_heads)))

        # RNN编码器
        # 之前是开启了文本编码，现在是位置编码
        position_embedding = PositionEmbedding(config.embedding_size).to(device)

        sp_model = SubjectPredicateModel(
            embedding_size=config.embedding_size,
            num_predicate=self.num_predicate,
            num_heads=config.num_heads,
            rnn_type=config.rnn_type,
            rnn_hidden_size=config.rnn_hidden_size,
            forward_dim=config.forward_dim,
            device=device
        ).to(device)

        o_model = ObjectModel(
            embedding_size=config.embedding_size,
            num_heads=config.num_heads,  # num_heads = 8
            rnn_type=config.rnn_type,
            rnn_hidden_size=config.rnn_hidden_size,
            forward_dim=config.forward_dim,
            device=device,
        ).to(device)

        # 滑动平均 处理 模型的参数
        sp_ema = EMA(model=sp_model, decay=0.999)
        o_ema = EMA(model=o_model, decay=0.999)

        # sp_model.apply(init_weights)
        # s_model.apply(init_weights)

        # reduction='none' 求出向量间对应元素loss后，不作求和或求平均的处理
        bce_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        # bce_loss =  DynamicFocalLossBCE(alpha=0.7, gamma=2.0, device=device).to(device)
        # bce_loss = FocalLossBCE(alpha=0.25, gamma=2.0, with_logits=True, device=device).to(device)
        f_bce_loss = FocalLossBCE(alpha=0.25, gamma=2.0, with_logits=True, device=device).to(device)
        # f_bce_loss = DynamicFocalLossBCE(alpha=0.25, gamma=2.0, device=device).to(device)

        # 网络参数
        params = []
        if config.from_pretrained not in ['bert', 'albert']:
            params.extend(get_models_parameters([embedding], weight_decay=0.0))
        params.extend(get_models_parameters(model_list=[sp_model, o_model], weight_decay=0.0))

        # 优化器
        optimizer = torch.optim.Adam(params=params, lr=config.learning_rate)

        # batch数 = 数据集样本数 / 一个batch的样本
        steps = int(np.round(self.train_data.len / config.batch_size))
        info = 'epoch: {}, steps: {}'.format(config.epoch, steps)
        print(info)

        best_f1 = 0.0
        best_epoch = 0

        patience = 10
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='max', # max表示当监控量停止上升的时候，学习率将减小
        #     factor=0.1, # new_lr = old_lr * factor
        #     patience=patience, # 容忍网路的性能不提升的次数，高于这个次数就降低学习率
        #     min_lr=0.0001,
        # )

        # warm_up_lambda是一个倍数算子
        warm_up_steps = int(steps * config.warm_up_epoch)
        warm_up_lambda = lambda step: (step + 1) / warm_up_steps
        warm_up_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_lambda)

        # 非等间隔余弦退火
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_T_max)

        # f1不上升的次数
        f1_not_up_count = 0
        f1 = 0.0
        loss_sum = 0.0
        loss_cpu = 0.0

        # 保存模型的名字
        model_name = '{}_wv{}_{}{}_{}'.format(config.from_pretrained, config.embedding_size, config.rnn_type,
                                              config.rnn_hidden_size, dataset)

        for epoch in range(config.epoch):
            sp_model.train()
            o_model.train()

            # 平均损失 = 总损失 / batch数
            log.info('epoch: {}, learning rate: {:.6f}, average batch loss: {:.6f}'.format(
                epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss_sum / steps
            )
            )
            loss_sum = 0.0

            # 进度条
            # 在with区间结束时，tqdm类会被回收，exit()
            with tqdm(total=steps) as pbar:
                # 拿到一个batch的train_data  step是序号
                # inputs_outputs: 一个 batch 的 collate_fn 的返回值
                for step, inputs_outputs in enumerate(train_data_loader):
                    pbar.update(1)
                    pbar.set_description('training epoch {}'.format(epoch))
                    pbar.set_postfix_str('loss: {:0.4f}'.format(loss_cpu))

                    # 将Tensor放到GPU上，计算loss的时候可以加速运算
                    text = inputs_outputs['text']
                    o_query_text = inputs_outputs['o_query_text']

                    sp_start_true = inputs_outputs['sp_start'].to(device)
                    sp_end_true = inputs_outputs['sp_end'].to(device)
                    o_start_true = inputs_outputs['o_start'].to(device)
                    o_end_true = inputs_outputs['o_end'].to(device)

                    # 字符转向量
                    if config.from_pretrained in ['bert', 'albert']:
                        input_embedding, cls_embedding, input_length, attention_mask = embedding(text)
                        o_query_embedding, cls_embedding, o_query_length, attention_mask = embedding(o_query_text)
                    else:
                        input_embedding, input_length = embedding(text, requires_grad=True)
                        o_query_embedding, o_query_length = embedding(o_query_text, requires_grad=True)

                    input_pos_embedding = position_embedding(input_length)
                    o_query_pos_embedding = position_embedding(o_query_length)

                    input_mask = create_mask_from_lengths(input_length)
                    query_mask = create_mask_from_lengths(o_query_length)

                    share_feature, sp_start_pred, sp_end_pred = sp_model(
                        input_embedding=input_embedding,
                        mask=input_mask,
                        position_embedding=input_pos_embedding,
                    )

                    o_start_pred, o_end_pred = o_model(
                        share_feature=share_feature,
                        share_mask=input_mask,
                        query_embedding=o_query_embedding,
                        query_mask=query_mask,
                        query_pos_embedding=o_query_pos_embedding,
                    )

                    # (batch_size, seq_len, 1) → (batch_size, seq_len）
                    o_start_pred = torch.squeeze(o_start_pred, dim=2)
                    o_end_pred = torch.squeeze(o_end_pred, dim=2)

                    # softmax
                    # o_start_pred = o_start_pred.permute(0, 2, 1)
                    # o_end_pred = o_end_pred.permute(0, 2, 1)

                    if config.from_pretrained in ['bert', 'albert']:
                        sp_start_pred = sp_start_pred[:, 1: - 1, :]
                        sp_end_pred = sp_end_pred[:, 1: - 1, :]
                        o_start_pred = o_start_pred[:, 1: - 1]
                        o_end_pred = o_end_pred[:, 1: - 1]
                        input_length -= 2
                        input_mask = create_mask_from_lengths(input_length)

                    # ge: greater than, >=
                    # 将sp_model的input_mask改造成True False形式 (32, 122)
                    loss_mask = torch.ge(input_mask, 1)
                    sp_loss_mask = loss_mask.unsqueeze(dim=2)  # (32,122,1)

                    # sp_start_loss = bce_loss(sp_start_pred, sp_start_true)
                    # # 将没有被masked的loss值压缩成一个1-DTensor
                    # sp_start_loss_masked = torch.masked_select(sp_start_loss, sp_loss_mask)

                    # 计算sp模型的损失， mask掉的损失不做处理, mask_select的输入规定要 boolean mask
                    sp_start_loss = torch.masked_select(bce_loss(sp_start_pred, sp_start_true), sp_loss_mask)
                    sp_end_loss = torch.masked_select(bce_loss(sp_end_pred, sp_end_true), sp_loss_mask)
                    sp_start_loss = torch.mean(sp_start_loss)
                    sp_end_loss = torch.mean(sp_end_loss)

                    # 计算o模型损失
                    o_start_loss = torch.masked_select(f_bce_loss(o_start_pred, o_start_true), loss_mask)
                    o_end_loss = torch.masked_select(f_bce_loss(o_end_pred, o_end_true), loss_mask)
                    o_start_loss = torch.mean(o_start_loss)
                    o_end_loss = torch.mean(o_end_loss)

                    # 计算总的损失
                    loss = Config.alpha * (sp_start_loss + sp_end_loss) + Config.beta * (o_start_loss + o_end_loss)
                    loss_cpu = loss.cpu().detach().numpy()
                    loss_sum += loss_cpu

                    optimizer.zero_grad()
                    # 计算Tensor：loss 的计算图的叶子节点的梯度
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters=sp_model.parameters(), max_norm=10, norm_type=2.0)
                    nn.utils.clip_grad_norm_(parameters=o_model.parameters(), max_norm=10, norm_type=2.0)
                    optimizer.step()

                    # apply ema
                    sp_ema.update_params()
                    o_ema.update_params()

                    if config.log_loss and (step % 100 == 0 or step == steps - 1):
                        log.info('epoch: {}, step: {}, loss: {:.5f}.'.format(epoch, step, loss_cpu))

                    # warm up
                    if epoch < config.warm_up_epoch:
                        warm_up_schedule.step()

                # end data loader
            # end process bar

            # Set the model in evaluation mode. Turn off dropout
            sp_model.eval()
            o_model.eval()

            # 应用滑动平均
            sp_ema.apply_shadow()
            o_ema.apply_shadow()

            print('{}, evaluate epoch: {} ...'.format(get_formated_time(), epoch))

            # Context-manager that disabled gradient calculation.
            with torch.no_grad():
                f1, precision, recall, (spo_list_pred, spo_list_true) = evaluate(
                    models=(sp_model, o_model),
                    embeddings=(embedding, position_embedding),
                    dev_data=self.test_data,
                    predicate_info=self.predicate_info,
                    config=config,
                    id2predicate=self.id2predicate,
                )

            restore_best_state = False
            if f1 >= best_f1:
                best_f1 = f1
                best_epoch = epoch
                f1_not_up_count = 0
                sp_ema.save_best_params()
                o_ema.save_best_params()
                # 记录测试集三元组预测
                save_spo_list(self.test_data, spo_list_pred, spo_list_true,
                              parent_path + '/data/' + dataset + '/spo_list_pred.json')
                if config.from_pretrained not in ['bert', 'albert']:
                    torch.save(embedding.state_dict(), '{}/{}_sp_o_embedding.pkl'.format(base_path, model_name))
                torch.save(sp_model.state_dict(), '{}/{}_sp_model.pkl'.format(base_path, model_name))
                torch.save(o_model.state_dict(), '{}/{}_o_model.pkl'.format(base_path, model_name))
            else:
                f1_not_up_count += 1
                if f1_not_up_count >= patience:
                    info = 'f1 do not increase {} times, restore best state...'.format(patience)
                    print_and_log(info, log)
                    sp_ema.restore_best_params()
                    o_ema.restore_best_params()
                    restore_best_state = True
                    f1_not_up_count = 0
                    # patience += 1

            if not restore_best_state:
                sp_ema.restore()
                o_ema.restore()

            info = 'epoch: {}, f1: {:.5f}, precision: {:.5f}, recall: {:.5f}, best_f1: {:.5f}, best_epoch: {}'.format(
                epoch, f1, precision, recall, best_f1, best_epoch)
            print(info)
            log.info(info)

            # 调整学习率
            # Note that step should be called after validate()
            if epoch > config.warm_up_epoch:
                lr_scheduler.step()