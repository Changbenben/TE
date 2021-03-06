import torch
import sys
import numpy as np
from config import Config

# from model.p_so_model import Trainer, load_model_and_test

# 选择 partial match / exact match
from model.sp_o import Trainer
from model.evaluation import load_model_and_test
# from model.sp_o_partial import Trainer, load_model_and_test

# from model.sp_o_model_v2 import Trainer, load_model_and_test
# from model.s_model import Trainer, load_model_and_test

seed = 233
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    # 设置默认为FloatTensor
    torch.set_default_tensor_type(torch.FloatTensor)

    # 加载配置文件
    config = Config()

    # 指定训练设备
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.cuda_device_number))
        torch.backends.cudnn.benchmark = True

    print('device: {}'.format(device))

    args = sys.argv
    mode = 'test'
    if len(args) >= 2:
        mode = args[1]

    print('mode: {}'.format(mode))

    if mode == 'train':
        trainer = Trainer()
        trainer.train(config, device)

    if mode == 'test':
        # 评估测试集的时候要关闭benchmark，否则会变慢
        torch.backends.cudnn.benchmark = False
        load_model_and_test(config, device)


