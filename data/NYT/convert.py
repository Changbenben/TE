import ujson
import numpy as np
from os.path import abspath, dirname

ID_PREDICATE_FILE = 'id_and_predicate_no_unk.json'
PREDICATE_INFO_FILE = 'predicate_info.json'
CHAR2ID_FILE = 'char2id.json'
NEW_CHAR2ID_FILE = 'char2id.json'
CHAR_ID2VEC = 'char_id2vector.json'
BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
save_base = BASE_DIR + '/model_file/pretrain'
save_numpy_file = save_base + '/NYT100.npy'


def generate_predicate_info():
    """
    生成predicate_info.json 为o_query做准备
    """
    with open(ID_PREDICATE_FILE, 'r', encoding='utf-8') as f, open(PREDICATE_INFO_FILE, 'w', encoding='utf-8') as f1:
        id_predicate = ujson.load(f)
        predicate_dict = dict()
        for key, value in id_predicate[0].items():
            types = value.split('/')
            temp = dict()
            temp['s_type'] = types[1]
            temp['p_type'] = types[3]
            temp['o_type'] = types[2]
            predicate_dict[value] = temp

        ujson.dump(predicate_dict, f1, indent=4, ensure_ascii=False)


def add_PAD():
    """
    在char2id.json 根据key排序
    手动：在新的char2id.json中 [PAD]=0  [UNK]:0→1  Hydro-Electric:1→190760
    """
    with open(CHAR2ID_FILE, 'r', encoding='utf-8') as f, open(NEW_CHAR2ID_FILE, 'w', encoding='utf-8') as f1:
        char2id = ujson.load(f)
        sorted_list = sorted(char2id.items(), key=lambda x: x[1], reverse=False)
        ujson.dump(dict(sorted_list), f1, indent=4, ensure_ascii=False)


def generate_pretrain_vector():
    # 制作word_vec: list(list) 90761*100
    # 在新的char2id.json中[PAD]=0   [UNK]:0→1   Hydro-Electric: 1→190760
    with open(CHAR_ID2VEC, 'r') as f:
        char_id2vec = ujson.load(f)

        # list[tuple]
        wordvec_list = []
        for _, wordvec in char_id2vec.items():
            wordvec_list.append(wordvec)

        # 调整位置
        wordvec_list.append(wordvec_list[1])
        wordvec_list[1] = wordvec_list[0]
        wordvec_list[0] = [0] * 100

    # 制作pretrain.npy
    embedding_init = np.array(wordvec_list)
    np.save(save_numpy_file, embedding_init)


if __name__ == '__main__':
    generate_pretrain_vector()
