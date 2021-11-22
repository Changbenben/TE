import sys

from embedding.torchEmbedding import TorchEmbedding, PositionEmbedding
from tqdm import tqdm
from config import Config
from utils.function import *
from utils.logger import Logger

dataset = Config.dataset

log = Logger('sp_o_{}'.format(dataset), std_out=False, save2file=True).get_logger()

parent_path = abspath(dirname(dirname(__file__)))
TRAIN_FILE = parent_path + '/data/' + dataset + '/my_train_data.json'
# DEV_FILE = parent_path + '/data/' + dataset + '/my_dev_data.json'
TEST_FILE = parent_path + '/data/' + dataset + '/my_test_data.json'
TEST_FILE_TYPE = parent_path + '/data/' + dataset + '/type_split/'
TEST_FILE_NUM = parent_path + '/data/' + dataset + '/num_split/'
TEST_FILE_TYPE_LIST = [TEST_FILE_TYPE + 'normal.json', TEST_FILE_TYPE + 'epo.json', TEST_FILE_TYPE + 'seo.json']
TEST_FILE_NUM_LIST = [TEST_FILE_NUM + 'triple_1.json', TEST_FILE_NUM + 'triple_2.json', TEST_FILE_NUM + 'triple_3.json',
                      TEST_FILE_NUM + 'triple_4.json', TEST_FILE_NUM + 'triple_5.json']

ID_PREDICATE_FILE = parent_path + '/data/' + dataset + '/id_and_predicate_no_unk.json'
PREDICATE_INFO_FILE = parent_path + '/data/' + dataset + '/predicate_info.json'
CHAR2ID_FILE = parent_path + '/data/' + dataset + '/char2id.json'


def evaluate(models: tuple, embeddings: tuple, dev_data: list, predicate_info: dict, config: Config, id2predicate: dict,
             show_details: bool = True):
    '''
    评估
    返回 f1, precision, recall, (spo_list_pred, spo_list_true)
    '''
    # 最小的评估batch是128，评估的batch_size，和训练的batch_size不同
    batch_size = config.batch_size if config.batch_size >= 128 else 128

    spo_list_true = []
    spo_list_pred = []
    batch_text = []
    n_dev_data = len(dev_data)

    # tqdm结合enumerate，两种写法，515行是第一种写法（可自定义）
    # 存够一个batch_size:128个样本，就进行一次compute_batch_spo
    for index, data in tqdm(enumerate(dev_data), total=n_dev_data):
        # 取出当前文本的真实spo,元素为一个tuple
        current_spo = []
        for spo in data['spo_list']:
            current_spo.append((spo['subject'], spo['predicate'], spo['object']))
        spo_list_true.append(current_spo)

        # 取出spo_list_true对应的text，用compute_batch_spo计算spo
        batch_text.append(data['text'])

        if len(batch_text) == batch_size or index == n_dev_data - 1:
            batch_spo = compute_batch_spo(
                models=models,
                embeddings=embeddings,
                text=batch_text,
                predicate_info=predicate_info,
                id2predicate=id2predicate,
                config=config,
            )
            spo_list_pred.extend(batch_spo)
            batch_text = []

    if show_details:
        def get_spo(spo_list_all: list):
            s_list, sp_list, p_list, o_list = [], [], [], []
            for spo_list in spo_list_all:
                current_s, current_sp, current_p, current_o = [], [], [], []
                for spo in spo_list:
                    current_s.append((spo[0],))
                    current_p.append((spo[1],))
                    current_o.append((spo[2],))
                    current_sp.append((spo[0], spo[1]))

                s_list.append(current_s)
                p_list.append(current_p)
                o_list.append(current_o)
                sp_list.append(current_sp)
            return s_list, p_list, o_list, sp_list

        def show_f1_p_r(p: list, t: list, name: str):
            f1, precision, recall = f1_p_r_compute(p, t, repair=False)
            info = '{}, f1: {:.5f}; precision: {:.5f}; recall: {:.5f} '.format(name, f1, precision, recall)
            print_and_log(info, log)

        s_pred, p_pred, o_pred, sp_pred = get_spo(spo_list_pred)
        s_true, p_true, o_true, sp_true = get_spo(spo_list_true)
        show_f1_p_r(s_pred, s_true, 'subject')
        show_f1_p_r(p_pred, p_true, 'preidcate')
        show_f1_p_r(o_pred, o_true, 'object')
        show_f1_p_r(sp_pred, sp_true, 'subject and predicate')

    f1, precision, recall = f1_p_r_compute(spo_list_pred, spo_list_true, repair=False)

    return f1, precision, recall, (spo_list_pred, spo_list_true)


def compute_batch_spo(models: tuple, embeddings: tuple, text: list, predicate_info: dict, id2predicate: dict,
                      config: Config):
    """
    接受一个batch
    """
    sp_model, o_model = models

    share_features, share_masks, sp_start_preds, sp_end_preds = compute_batch_sp(
        sp_model=sp_model,
        embeddings=embeddings,
        text=text,
        config=config,
    )

    batch_size = share_features.shape[0]
    batch_sp = []
    batch_share_feature = []
    batch_mask = []
    batch_o_query_text = []
    batch_ids = []

    # 抽取一个batch的sp
    # zip，每一个样本的share_feature, share_mask, sp_start_pred, sp_end_pred组成一个元组
    # 返回元组列表,元素个数为batch_size
    for bs_id, (share_feature, share_mask, sp_start_pred, sp_end_pred) in enumerate(
            zip(share_features, share_masks, sp_start_preds, sp_end_preds)):
        text_ = text[bs_id]
        token_list = text_.strip().split(' ')
        # 从这句话每个字符中抽取
        # 返回大于0.4 的元素的坐标，一种坐标组成一个元组
        sp_start = np.where(sp_start_pred[0: len(token_list)] >= Config.threshold['sp_start'])
        sp_end = np.where(sp_end_pred[0: len(token_list)] >= Config.threshold['sp_end'])

        # 取横坐标和纵坐标：s_start位置 和 p的索引
        # 因为for包括s也包括p，所以每一个p都会考虑到，并根据p生成o_query_text
        for s_start, s_p_id in zip(*sp_start):
            for s_end, e_p_id in zip(*sp_end):
                if s_start <= s_end and s_p_id == e_p_id:
                    # 发现一个sp
                    subject = " ".join(token_list[s_start: s_end + 1])
                    predicate = id2predicate[str(s_p_id)]

                    p_info = predicate_info[predicate]
                    o_query_text = '{}，{}，{}，{}。{}'.format(p_info['s_type'], subject, predicate, p_info['o_type'],
                                                           text_)

                    batch_ids.append(bs_id)  # 记录sp是那一条text的
                    batch_sp.append((subject, predicate))
                    batch_share_feature.append(share_feature)
                    batch_mask.append(share_mask)
                    batch_o_query_text.append(o_query_text)

                    # 就近匹配，不再往下找了
                    break

    last_start = 0
    # ceil(sp数量/batch数量)
    n = int(np.ceil(len(batch_o_query_text) / batch_size))
    batch_spo_pred = [[] for _ in range(batch_size)]

    max_seq_len = share_features.shape[1]
    embedding_dim = share_features.shape[2]

    # 对一个batch中抽取到的所有sp抽取o
    for _ in range(n):
        end = last_start + batch_size
        # 输入模型需要Tensor，但之前每个样本是组成了list，所以要转成Tensor
        # 取一个batch 的 batch_share_feature
        # 直接cat结果不对，要重新reshape
        bs_share_feature = torch.cat(batch_share_feature[last_start: end], dim=0)
        bs_share_feature = bs_share_feature.reshape(-1, max_seq_len, embedding_dim)

        bs_input_mask = batch_mask[last_start: end]
        bs_input_mask = torch.cat(bs_input_mask, dim=0)
        bs_input_mask = bs_input_mask.reshape(-1, max_seq_len)

        o_start_preds, o_end_preds = compute_o(
            o_model=o_model,
            embeddings=embeddings,
            share_feature=bs_share_feature,
            share_mask=bs_input_mask,
            o_query_text=batch_o_query_text[last_start: end],
            config=config,
        )

        # zip得到元组列表 for tuple in zip() tuple代表一个样本这些数据组成的元组，如果tuple写成多个变量，则可以将元组拆开
        for bs_id, sp, o_start_pred, o_end_pred in zip(batch_ids[last_start: end], batch_sp[last_start: end],
                                                       o_start_preds, o_end_preds):

            # 可能有多个o
            text_ = text[bs_id]
            token_list = text_.strip().split(' ')

            o_start = o_start_pred[0: len(token_list)]
            o_end = o_end_pred[0: len(token_list)]

            for i, o_s in enumerate(o_start):
                if o_s >= Config.threshold['o_start']:
                    for j, o_e in enumerate(o_end[i:]):
                        if o_e >= Config.threshold['o_end']:
                            object_ = " ".join(token_list[i: i + j + 1])
                            batch_spo_pred[bs_id].append(sp + (object_,))
                            break
        # 更新last start
        last_start += batch_size

    return batch_spo_pred


def compute_batch_sp(sp_model, embeddings: tuple, text: list, config: Config):
    '''
    计算一个batch的dev_data的sp_model结果
    返回share_feature, input_mask, sp_start_pred, sp_end_pred
    '''
    embedding, position_embedding = embeddings
    sigmoid = torch.sigmoid

    if config.from_pretrained in ['bert', 'albert']:
        input_embedding, cls_embedding, input_length, attention_mask = embedding(text)
    else:
        input_embedding, input_length = embedding(text)
    input_pos_embedding = position_embedding(input_length)
    input_mask = create_mask_from_lengths(input_length)

    share_feature, sp_start_pred, sp_end_pred = sp_model(
        input_embedding=input_embedding,
        mask=input_mask,
        position_embedding=input_pos_embedding,
    )

    if config.from_pretrained in ['bert', 'albert']:
        sp_start_pred = sp_start_pred[:, 1: - 1, :]
        sp_end_pred = sp_end_pred[:, 1: - 1, :]

    sp_start_pred = sigmoid(sp_start_pred).cpu().detach().numpy()
    sp_end_pred = sigmoid(sp_end_pred).cpu().detach().numpy()

    return share_feature, input_mask, sp_start_pred, sp_end_pred


def compute_o(o_model, embeddings: tuple, share_feature: Tensor, share_mask: Tensor, o_query_text: list,
              config: Config):
    '''
    '''
    embedding, position_embedding = embeddings
    sigmoid = torch.sigmoid

    if config.from_pretrained in ['bert', 'albert']:
        o_query_embedding, cls_embedding, o_query_length, attention_mask = embedding(o_query_text)
    else:
        o_query_embedding, o_query_length = embedding(o_query_text)

    o_query_pos_embedding = position_embedding(o_query_length)
    query_mask = create_mask_from_lengths(o_query_length)

    o_start_pred, o_end_pred = o_model(
        share_feature=share_feature,
        share_mask=share_mask,
        query_embedding=o_query_embedding,
        query_mask=query_mask,
        query_pos_embedding=o_query_pos_embedding,
    )
    # sigmoid
    o_start_pred = torch.squeeze(o_start_pred, dim=2)
    o_end_pred = torch.squeeze(o_end_pred, dim=2)

    if config.from_pretrained in ['bert', 'albert']:
        o_start_pred = o_start_pred[:, 1: - 1]
        o_end_pred = o_end_pred[:, 1: - 1]

    o_start_pred = sigmoid(o_start_pred).cpu().detach().numpy()
    o_end_pred = sigmoid(o_end_pred).cpu().detach().numpy()
    return o_start_pred, o_end_pred


def load_model_and_test(config: Config, device):
    from model.sp_o import SubjectPredicateModel, ObjectModel
    base_path = parent_path + '/model_file'
    dev_data = read_json(TEST_FILE)
    dev_data_type = [read_json(FILE) for FILE in TEST_FILE_TYPE_LIST]
    dev_data_num = [read_json(FILE) for FILE in TEST_FILE_NUM_LIST]

    embedding = TorchEmbedding(config.embedding_size, device).to(device)
    position_embedding = PositionEmbedding(config.embedding_size).to(device)

    id2predicate, predicate2id = read_json(ID_PREDICATE_FILE)
    num_predicate = len(id2predicate)
    predicate_info = read_json(PREDICATE_INFO_FILE)

    sp_model = SubjectPredicateModel(
        embedding_size=config.embedding_size,
        num_predicate=num_predicate,
        num_heads=config.num_heads,
        rnn_type=config.rnn_type,
        rnn_hidden_size=config.rnn_hidden_size,
        forward_dim=config.forward_dim,
        device=device
    ).to(device)

    o_model = ObjectModel(
        embedding_size=config.embedding_size,
        num_heads=config.num_heads,
        rnn_type=config.rnn_type,
        rnn_hidden_size=config.rnn_hidden_size,
        forward_dim=config.forward_dim,
        device=device,
    ).to(device)

    model_name = '{}_wv{}_{}{}_{}'.format(config.from_pretrained, config.embedding_size, config.rnn_type,
                                          config.rnn_hidden_size, dataset)

    embedding.load_state_dict(
        torch.load('{}/{}_sp_o_embedding.pkl'.format(base_path, model_name),
                   map_location=device))
    sp_model.load_state_dict(torch.load('{}/{}_sp_model.pkl'.format(base_path, model_name), map_location=device))
    o_model.load_state_dict(torch.load('{}/{}_o_model.pkl'.format(base_path, model_name), map_location=device))

    sp_model.eval()
    o_model.eval()

    # 完整 test
    with torch.no_grad():
        f1, precision, recall, (spo_list_pred, spo_list_true) = evaluate(
            models=(sp_model, o_model),
            embeddings=(embedding, position_embedding),
            dev_data=dev_data,
            predicate_info=predicate_info,
            config=config,
            id2predicate=id2predicate,
            show_details=True,
        )
    save_spo_list(dev_data, spo_list_pred, spo_list_true, parent_path + '/data/' + dataset + '/spo_list_pred.json')
    print('f1: {:.5f}; precision: {:.5f}; recall: {:.5f}'.format(f1, precision, recall))

    # split_by_type
    print('Test result for type split')
    type_f1 = []
    for dev_data in dev_data_type:
        f1, precision, recall, (spo_list_pred, spo_list_true) = evaluate(
            models=(sp_model, o_model),
            embeddings=(embedding, position_embedding),
            dev_data=dev_data,
            predicate_info=predicate_info,
            config=config,
            id2predicate=id2predicate,
            show_details=True,
        )
        type_f1.append(f1)
        print('f1: {:.5f}; precision: {:.5f}; recall: {:.5f}'.format(f1, precision, recall))
    print(type_f1)

    # split_by_num
    print('Test result for num split')
    num_f1 = []
    for dev_data in dev_data_num:
        f1, precision, recall, (spo_list_pred, spo_list_true) = evaluate(
            models=(sp_model, o_model),
            embeddings=(embedding, position_embedding),
            dev_data=dev_data,
            predicate_info=predicate_info,
            config=config,
            id2predicate=id2predicate,
            show_details=True,
        )
        num_f1.append(f1)
        print('f1: {:.5f}; precision: {:.5f}; recall: {:.5f}'.format(f1, precision, recall))
    print(num_f1)


def load_model_and_test200(config: Config, device):
    import time
    from model.sp_o import SubjectPredicateModel, ObjectModel

    base_path = parent_path + '/model_file'
    dev_data = read_json(TEST_FILE)[:200]

    embedding = TorchEmbedding(config.embedding_size, device).to(device)

    id2predicate, predicate2id = read_json(ID_PREDICATE_FILE)
    num_predicate = len(id2predicate)
    predicate_info = read_json(PREDICATE_INFO_FILE)

    position_embedding = PositionEmbedding(config.embedding_size).to(device)

    sp_model = SubjectPredicateModel(
        embedding_size=config.embedding_size,
        num_predicate=num_predicate,
        num_heads=config.num_heads,
        rnn_type=config.rnn_type,
        rnn_hidden_size=config.rnn_hidden_size,
        forward_dim=config.forward_dim,
        device=device
    ).to(device)

    o_model = ObjectModel(
        embedding_size=config.embedding_size,
        num_heads=config.num_heads,
        rnn_type=config.rnn_type,
        rnn_hidden_size=config.rnn_hidden_size,
        forward_dim=config.forward_dim,
        device=device,
    ).to(device)

    model_name = '{}_wv{}_{}{}'.format(config.from_pretrained, config.embedding_size, config.rnn_type,
                                       config.rnn_hidden_size)

    embedding.load_state_dict(
        torch.load('{}/{}_sp_o_embedding.pkl'.format(base_path, model_name), map_location='cuda:0'))
    sp_model.load_state_dict(torch.load('{}/{}_sp_model.pkl'.format(base_path, model_name), map_location='cuda:0'))
    o_model.load_state_dict(torch.load('{}/{}_o_model.pkl'.format(base_path, model_name), map_location='cuda:0'))

    sp_model.eval()
    o_model.eval()

    time_start = time.time()
    with torch.no_grad():
        f1, precision, recall, (spo_list_pred, spo_list_true) = evaluate(
            models=(sp_model, o_model),
            embeddings=(embedding, position_embedding),
            dev_data=dev_data,
            predicate_info=predicate_info,
            config=config,
            id2predicate=id2predicate,
            show_details=True,
        )
    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))
    save_spo_list(dev_data, spo_list_pred, spo_list_true, parent_path + '/data/spo_list_pred.json')
    print('f1: {:.5f}; precision: {:.5f}; recall: {:.5f}\n'.format(f1, precision, recall))
