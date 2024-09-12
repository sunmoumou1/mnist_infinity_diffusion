# 重新复习

from ml_collections import ConfigDict # type: ignore
from ml_collections.config_dict import FieldReference # type: ignore

# loss_type只能选择mse或者l1

def get_config():
    config = ConfigDict()

    config.run = run = ConfigDict()
    run.name = 'infty_diff_era5_wo_encoder' # name是用来给wandb区分不同实验的
    run.experiment = 'q_mollified_240_121' # experiment是用来保存checkpoints的
    run.wandb_dir = ''
    run.wandb_mode = 'online'

    config.data = data = ConfigDict()
    data.name = 'era5_specific_humidity'
    data.root_dir = "/home/shepherdchinacan/era5-infinity-diffusion_wo_encoder/process_era5_data_before/specific_humidity_1973_2017.npy"
    data.img_size = (FieldReference(240), FieldReference(121))
    data.channels = 1
    data.fid_samples = 100 # 在生成样本的时候才起作用
    data.start_year = 1973
    data.end_year = 2017
    
    # 获取原始的图像尺寸
    img_height = config.data.img_size[0]._value
    img_width = config.data.img_size[1]._value
   
    config.train = train = ConfigDict()
    train.load_checkpoint = False
    train.amp = True
    train.batch_size = 40
    train.sample_size = 4
    train.plot_graph_steps = 100
    train.plot_samples_steps = 500
    train.checkpoint_steps = 2000
    train.ema_update_every = 1
    train.ema_decay = 0.999

    config.model = model = ConfigDict()
    model.nf = 64
    model.time_emb_dim = 128
    model.num_conv_blocks = 3
    model.knn_neighbours = 3
    model.depthwise_sparse = True  # 表示是否使用稀疏深度卷积，作用于SparseConvResBlock
    model.kernel_size = 5
    model.backend = "torchsparse"
    model.uno_res = (128, 64)
    model.uno_base_channels = 64
    model.uno_mults = (1,2,4,8)
    model.uno_blocks_per_level = (2,2,2,2)
    model.uno_attn_resolutions = [16,8]
    model.uno_dropout_from_resolution = 16
    model.uno_dropout = 0.1
    model.uno_conv_type = "conv"
    model.z_dim = 256
    model.sigma_small = True
    model.stochastic_encoding = False
    model.kld_weight = 1e-4

    config.diffusion = diffusion = ConfigDict()
    diffusion.steps = 100
    diffusion.noise_schedule = 'cosine'
    diffusion.schedule_sampler = 'uniform' # 可选择的参数为loss-second-moment, uniform，github上默认使用uniform
    diffusion.loss_type = 'L1'
    diffusion.gaussian_filter_std = 1.0
    diffusion.model_mean_type = "mollified_epsilon"
    diffusion.multiscale_loss = False
    diffusion.multiscale_max_img_size = (img_height // 2, img_width // 2) # 这个参数究竟是什么含义来着？
    diffusion.mollifier_type = "dct" 
    # 另外还可以选择conv的方式，但是这种方式没有使用，而且原作者在这里搞了一个半吊子工程，这个conv的方式完全不可以使用，虽然通过这个方式创建的self.mollifier可以实现高斯滤波，但是它完全没有办法实现self.mollifier.undo_wiener(),因为完全不可以使用

    config.mc_integral = mc_integral = ConfigDict()
    mc_integral.type = 'uniform'
    mc_integral.q_sample = int( (img_height * img_width) // 2 )
    # 注意mc_integral.q_sample很重要，这个参数控制采样的像素点数量

    config.optimizer = optimizer = ConfigDict()
    optimizer.learning_rate = 1e-4
    optimizer.adam_beta1 = 0.9
    optimizer.adam_beta2 = 0.99
    optimizer.warmup_steps = 500 # 在原github上这里的warmup_steps是0
    optimizer.gradient_skip = True
    optimizer.gradient_skip_threshold = 500.

    return config


config = get_config()