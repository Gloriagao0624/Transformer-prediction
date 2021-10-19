import argparse
from .templates import set_template
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS

parser = argparse.ArgumentParser('训练架构全体配置参数')

################
# Top Level
################
parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'analyse'], help='程序的运行模式')
parser.add_argument('--template', type=str, default='debug', help='template.py中选择何种配置')

################
# 训练数据
################
# dataset #
parser.add_argument('--trainset_path',
                    type=str,
                    default='/home/oppoer/work/app_data_PDEM10_20210312/data.train.101',
                    help='训练集路径')
parser.add_argument('--valset_path',
                    type=str,
                    default='/home/oppoer/work/app_data_PDEM10_20210312/data.val.101',
                    help='验证集路径')
parser.add_argument('--testset_path',
                    type=str,
                    default='/home/oppoer/work/app_data_PDEM10_20210312/data.test.101',
                    help='测试集路径')
parser.add_argument('--train_app_stat_path',
                    type=str,
                    default='/home/oppoer/work/app_data_PDEM10_20210312/user_app.stat.freq.train',
                    help='训练集用户保留列表')
parser.add_argument('--val_app_stat_path',
                    type=str,
                    default='/home/oppoer/work/app_data_PDEM10_20210312/user_app.stat.freq.val',
                    help='验证集用户保留列表')
parser.add_argument('--test_app_stat_path',
                    type=str,
                    default='/home/oppoer/work/app_data_PDEM10_20210312/user_app.stat.freq.test',
                    help='测试集用户保留列表')
parser.add_argument('--longtail_path',
                    type=str,
                    default='/home/oppoer/work/app_data_PDEM10_20210312/longtail.5000',
                    help='用户的长尾App列表')
parser.add_argument('--user_embed_path', type=str, help='测试集用户保留列表')
parser.add_argument('--dataset_wechat_ratio', type=str, default=0.4572, help='数据集中微信的占比')
parser.add_argument('--from_memory', type=str, default=False, help='是否将数据集一次性加载到内存，慎重True，可能内存不够导致容器崩溃')
parser.add_argument('--class_num', type=int, default=5000, help='分类数')
parser.add_argument('--series_len', type=int, default=101, help='模型的输入样本长度')
# dataloader #
parser.add_argument('--train_batch_size', type=int, default=128, help='训练集Batch Size')
parser.add_argument('--val_batch_size', type=int, default=128, help='验证集Batch Size')
parser.add_argument('--test_batch_size', type=int, default=256, help='测试集Batch Size')
parser.add_argument('--dataloader_random_seed', type=float, default=10086, help='随机负样本构造种子数')

################
# 训练器
################
parser.add_argument('--trainer_code', type=str, default='sas', choices=TRAINERS.keys())
parser.add_argument('--resume_training', type=str, default=False)
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--enable_lr_schedule', default=False)
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=10, help='每隔多少Batch_size打印一次log')
# evaluation #
parser.add_argument('--metric_name', nargs='+', type=int, default=['Acc'], help='Metric 的名字')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 2, 4], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='Acc@4', help='确定最佳模型的度量')
parser.add_argument('--ignore_class', nargs='+', type=int, default=None, help='不进行分类排序的类别')

parser.add_argument('--dropout_p', type=float, default=None, help='Dropout 失活比例')

################
# 特征抽取层模型
################
parser.add_argument('--model_code', type=str, default='sas', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=float, default=0, help='固定模型训练的随机种子')
parser.add_argument('--item_embed_dim', type=int, default=None, help='item 隐向量的大小')
# SAS #
parser.add_argument('--d_model', type=int, default=None, help='模型隐向量的大小 (d_model)')
parser.add_argument('--sas_num_blocks', type=int, default=None, help='transformer encoder layers 层数')
parser.add_argument('--sas_num_heads', type=int, default=None, help='Number of heads for multi-attention')

# RNN #
parser.add_argument('--split_block', type=int, default=None, help='RNN 的序列输入是否分块')

################
# 特征交叉层模型，任务层
################
parser.add_argument('--task_inputs_series_len', type=int, default=None, help='取序列最后x个item进MLP，输入维度基于此计算')
parser.add_argument('--task_inputs_dim', nargs='+', type=int, default=None, help='任务层MLP网络的输入维度，可选')
parser.add_argument('--task_hidden_units', nargs='+', type=int, default=None, help='任务层MLP网络的隐维度')
parser.add_argument('--activation', type=str, default='relu', help='网络的激活层')
parser.add_argument('--multiclass', type=str, default='multiclass', help='任务层的任务，[multiclass, binary]')

################
# 实验数据
################
parser.add_argument('--experiment_dir', type=str, default='experiments', help='实验数据存放的根目录名称')
parser.add_argument('--experiment_description', type=str, default='train', help='trainer 侧实验数据存放文件夹的描述')
parser.add_argument('--experiment_import_root', type=str, default='test', help='tester 侧实验数据存放文件夹的描述')

args = parser.parse_args()
set_template(args)
