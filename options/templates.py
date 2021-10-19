def set_template(args):

    if args.template is None:
        return

    elif args.template.startswith('sas'):
        args.run_mode = 'train'  # train 、 analyse
        args.resume_training = False
        args.experiment_description = 'c3timesas_six_([64]-[4-2]-[1-5000])-(s-h)'

        # 数据集侧
        args.class_num = 5001
        args.series_len = 101

        # DataLoader 侧
        args.dataloader_code = 'sas2_series|time'  # sas_series|time 、 sas_series 、 sas2_series|time
        batch = 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer/Tester 侧
        args.trainer_code = 'sas'
        args.device_idx = '0'

        args.num_epochs = 14
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        args.ignore_class = [0]

        # 模型侧
        args.model_code = 'embed|cnn|timesas|mlp'  # embed|cnn|timesas|mlp, embed|sas|mlp、embed|cnn|sas|mlp 、 embed|timesas|mlp
        args.dropout_p = 0.1

        args.item_embed_dim = 64
        args.d_model = args.item_embed_dim
        args.sas_num_blocks = 4
        args.sas_num_heads = 2

        args.task_inputs_series_len = 1
        args.task_hidden_units = [args.class_num]
        args.activation = 'relu'
        args.task = 'multiclass'

    elif args.template.startswith('sas500'):
        args.run_mode = 'train'  # train 、 analyse
        args.resume_training = False
        # c3sas_svm_new101_([64]-[4-2]-[1-5000])-(s)-cls_2021-03-15_1
        # c3timesas_svm_new101_([64]-[4-2]-[1-5000])-(s-h)-cls_2021-03-15_0
        args.experiment_description = 'c3timesas_500_([64]-[4-2]-[1-5000])-(s-h-dt-dura)'

        # 数据集侧
        args.class_num = 5001
        args.series_len = 101

        # DataLoader 侧
        args.dataloader_code = 'sas2_series|time'  # sas_series|time 、 sas_series 、 sas2_series|time
        batch = 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer/Tester 侧
        args.trainer_code = 'sas'
        args.device_idx = '0'

        args.num_epochs = 14
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        args.ignore_class = [0]

        # 模型侧
        args.model_code = 'embed|cnn|timesas|mlp'  # embed|cnn|timesas|mlp, embed|sas|mlp、embed|cnn|sas|mlp
        args.dropout_p = 0.1

        args.item_embed_dim = 64
        args.d_model = args.item_embed_dim
        args.sas_num_blocks = 4
        args.sas_num_heads = 2

        args.task_inputs_series_len = 1
        args.task_hidden_units = [args.class_num]
        args.activation = 'relu'
        args.task = 'multiclass'

    elif args.template.startswith('mlp'):
        args.run_mode = 'train'
        args.resume_training = False
        args.experiment_description = 'mlp_svm4_101_([64]-[640-5000])-(s)-cls'

        # 数据集侧
        args.class_num = 5000
        args.series_len = 101

        # DataLoader 侧
        args.dataloader_code = 'series'
        batch = 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer 侧
        args.trainer_code = 'e2e'
        args.device_idx = '1'

        args.num_epochs = 14
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        args.ignore_class = [0]

        # 模型侧
        args.model_code = 'embed|mlp'
        args.dropout_p = 0.1

        args.item_embed_dim = 64
        args.task_inputs_series_len = 10
        args.task_hidden_units = [args.class_num]
        args.activation = 'relu'
        args.task = 'multiclass'

    elif args.template.startswith('tfm'):
        args.run_mode = 'train'
        args.resume_training = False
        args.experiment_description = 'timerestfm_six_([64]-[4-2]-[2624-64-5000])'

        # 数据集侧
        args.class_num = 5001
        args.series_len = 101

        args.dataloader_code = 'sas2_series|time'  # sas_series|time 、 sas_series 、 sas2_series|time
        batch = 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer 侧
        args.trainer_code = 'e2e'
        args.device_idx = '1'

        args.num_epochs = 14
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        args.ignore_class = [0]

        # 模型侧
        args.model_code = 'super_long_model'
        args.dropout_p = 0.1

        args.item_embed_dim = 64
        args.d_model = args.item_embed_dim
        args.sas_num_blocks = 4
        args.sas_num_heads = 2

        args.task_inputs_dim = 2624
        args.task_hidden_units = [64, args.class_num]
        args.activation = 'relu'
        args.task = 'multiclass'

    elif args.template.startswith('rnn_30w'):
        args.run_mode = 'train'
        args.resume_training = False
        args.experiment_description = 'rnn_find30w-task=9-split100-64-cls5000'

        # Dataset 侧
        args.class_num = 5000
        args.series_len = 101

        # DataLoader 侧
        batch = 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer 侧
        args.trainer_code = 'e2e'
        args.device_idx = '1'

        args.num_epochs = 7
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        args.ignore_class = [0]

        # 模型侧
        args.model_code = 'embed|rnn|mlp'
        args.dropout_p = 0.1

        args.item_embed_dim = 64
        args.split_block = 100

        args.task_inputs_series_len = 9
        args.task_hidden_units = [512, args.class_num]
        args.activation = 'relu'
        args.task = 'multiclass'

    elif args.template.startswith('mnist'):

        args.experiment_description = 'mnist'
        args.resume_training = False

        # DataLoader 侧
        args.dataloader_code = 'mnist'
        batch = 64
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer 侧
        args.trainer_code = 'mnist'
        args.device_idx = '0'
        args.device = 'cuda'

        args.num_epochs = 2
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        # 模型侧
        args.model_code = 'mnist'

    elif args.template.startswith('usas_30w'):
        args.run_mode = 'train'
        args.resume_training = False
        args.experiment_description = 'debug123'

        # 数据集侧
        args.class_num = 5000
        args.series_len = 101

        args.valset_path = '/home/oppoer/work/app_data_PDEM10_20210118/data.val.101.hg'
        args.testset_path = '/home/oppoer/work/app_data_PDEM10_20210118/data.test.101.hg'
        args.user_embed_path = '/home/oppoer/work/app_data_PDEM10_20210118/stack_embedding_train_50.json'

        # DataLoader 侧
        args.dataloader_code = 'uembed|series'  # all_feature 、 series
        batch = 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer/Tester 侧
        args.trainer_code = 'sas'
        args.device_idx = '1'

        args.num_epochs = 14
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        args.ignore_class = [0]

        # 模型侧
        args.model_code = 'embed|usas|mlp'
        args.dropout_p = 0.1

        args.item_embed_dim = 64
        args.d_model = args.item_embed_dim
        args.sas_num_blocks = 4
        args.sas_num_heads = 2

        args.task_inputs_series_len = 1
        args.task_hidden_units = [512, args.class_num]
        args.activation = 'relu'
        args.task = 'multiclass'

    elif args.template.startswith('debug'):
        args.run_mode = 'train'  # train 、 analyse
        args.resume_training = False
        args.experiment_dir = 'debugs'
        args.experiment_description = '123321'

        # 数据集侧
        args.class_num = 5001
        args.series_len = 101

        args.trainset_path = '/home/oppoer/work/app_data_PDEM10_20210312/data.train.101.small'
        args.testset_path = '/home/oppoer/work/app_data_PDEM10_20210312/data.val.101'

        # DataLoader 侧
        args.dataloader_code = 'sas2_series|time'  # sas_series|time 、 sas_series 、 sas2_series|time
        batch = 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #  Trainer/Tester 侧
        args.trainer_code = 'sas'
        args.device_idx = '0'

        args.num_epochs = 14
        args.metric_ks = [1, 2, 3, 4]
        args.best_metric = 'Acc@4'
        args.ignore_class = [0]

        # 模型侧
        args.model_code = 'embed|timesas|mlp'  # embed|cnn|timesas|mlp, embed|sas|mlp、embed|cnn|sas|mlp
        args.dropout_p = 0.1

        args.item_embed_dim = 64
        args.d_model = args.item_embed_dim
        args.sas_num_blocks = 4
        args.sas_num_heads = 2

        args.task_inputs_series_len = 1
        args.task_hidden_units = [args.class_num]
        args.activation = 'relu'
        args.task = 'multiclass'