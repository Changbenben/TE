from dataclasses import dataclass
import platform


@dataclass
class Config(object):
    dataset = 'NYT'
    epoch = 200
    batch_size = 32 if platform.system() == 'Windows' else 64
    learning_rate = 0.001

    # 生成批处理数据时的进程数量
    num_workers = 0 if platform.system() == 'Windows' else 1

    # 是否将loss保存到文件
    log_loss = False

    # 最后一个epoch的学习率衰减为初始学习率的 1 / 10 （大约）
    lr_T_max = int(epoch * 0.1)
    if dataset == 'WebNLG' or 'NYT':
        embedding_size = 100
        # MutilHeadAttention / SelfAtttention heads
        # 注意力头数必须能被词向量维度整除，embedding_size % num_heads === 0
        num_heads = 10
        rnn_hidden_size = 200
    else:
        embedding_size = 128
        num_heads = 8
        rnn_hidden_size = 256

    cuda_device_number = 0

    # 训练时，用前多少个epoch做warm up
    warm_up_epoch = 1

    # output linear forward dim
    # forward_dim = int(embedding_size * 2)
    forward_dim = 256
    # rnn_type = ['gru', 'lstm']
    rnn_type = 'gru'

    assert embedding_size % num_heads == 0

    # 预训练词向量，选项为['none','word2vec', 'albert', 'bert']
    # 对应的词向量为: [embedding_size, 300, 768, 768]
    from_pretrained = 'NYT'

    # 阈值
    threshold = {
        'sp_start': 0.35,
        'sp_end': 0.35,
        'o_start': 0.4,
        'o_end': 0.35
    }

    # loss系数
    alpha = 2.5
    beta = 1.0

    # bert
    bert_forward_dim = 256
    # 'cpu' or 'cuda'/'cuda:0'
    bert_device = 'cuda:0'

    # legacy:
    predicate_embedding_szie = 64
    sigmoid_threshold = 0.5
