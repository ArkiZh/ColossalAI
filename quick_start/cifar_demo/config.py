# 加载配置文件后可以通过colossalai.core.global_context来获取配置文件中的变量。

BATCH_SIZE = 512
NUM_EPOCHS = 2

# ============================= 混合精度 =================================
# 配置混合精度训练
from colossalai.amp import AMP_TYPE


# 使用 Torch AMP
fp16=dict(
    mode=AMP_TYPE.TORCH,

    # 下列是grad scaler的默认值
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True
)


# # 使用 naive AMP
# fp16 = dict(
#     mode=AMP_TYPE.NAIVE,

#     # below are the default values
#     # log_num_zeros_in_grad=False,
#     initial_scale=2 ** 32,
#     min_scale=1,
#     growth_factor=2,
#     backoff_factor=0.5,
#     growth_interval=1000,
#     hysteresis=2
# )

# # 使用 Nvidia Apex AMP
# fp16 = dict(
#     mode=AMP_TYPE.APEX,

#     # 下列是默认值
#     enabled=True,
#     opt_level='O1',
#     cast_model_type=None,
#     patch_torch_functions=None,
#     keep_batchnorm_fp32=None,
#     master_weights=None,
#     loss_scale=None,
#     cast_model_outputs=None,
#     num_losses=1,
#     verbosity=1,
#     min_loss_scale=None,
#     max_loss_scale=16777216.0
# )

# ============================= 梯度累积 =================================
# 梯度累积，整数值代表期望梯度累积的次数
gradient_accumulation = 4

# ============================= 梯度剪裁 =================================
# 梯度剪裁，建议在这设置而不是自己编写，因为原生的梯度剪裁在应用张量并行、流水线并行、MoE 等功能时可能会失败
clip_grad_norm = 1.0

# 基于Chunk内存管理的零冗余优化器 (ZeRO)
# 需要在代码里编写


# ============================= 张量并行 =================================
# 并行度要求：GPUs = pipeline parallel size x tensor parallel size x data parallel size
# 1d张量并行  在2个GPU上
parallel=dict(
    data=2,
    pipeline=1,
    tensor=dict(size=1, mode='1d'),
)

# # 2d张量并行  在4个GPU上
# parallel=dict(
#     data=1,
#     pipeline=1,
#     tensor=dict(size=4, mode='2d'),
# )
#
# # 2.5D张量并行，在8个 GPU 上
# parallel=dict(
#     data=1,
#     pipeline=1,
#     tensor=dict(size=8, mode='2.5d', depth=2),
# )
#
# # 3D张量并行，在8个 GPU 上
# parallel=dict(
#     data=1,
#     pipeline=1,
#     tensor=dict(size=8, mode='3d'),
# )

# ============================= 流水线并行 =================================
# 这种模型需要在代码里手动把模型拆分成一系列stage，不便统一配置

# NUM_MICRO_BATCHES=4
# parallel=dict(pipeline=2)


# ============================= NVMe offload =================================
# 为 Adam (CPUAdam 和 HybridAdam) 实现了优化器状态的 NVMe offload。
# 需要在代码里写，不便统一配置
