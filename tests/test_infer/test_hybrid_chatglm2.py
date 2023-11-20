import importlib.util

import pytest
import torch
import torch.distributed as dist
from packaging import version

import colossalai
from colossalai.inference import CaiInferEngine
from colossalai.shardformer.modeling.chatglm2_6b.configuration_chatglm import ChatGLMConfig
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")
HAS_LIGHTLLM_KERNEL = True

if importlib.util.find_spec("lightllm") is None:
    HAS_LIGHTLLM_KERNEL = False


def data_gen():
    input_ids = torch.tensor([[15496, 11, 616, 3290, 318, 13779, 318, 13779]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


inputs = data_gen()
for k, v in inputs.items():
    if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
        new_shape = [1] * v.dim()
        new_shape[0] = 16
        inputs[k] = v.to("cuda").repeat(*new_shape)


def pipeline_inference_test(tp_size, pp_size, max_output_len, micro_batch_size):
    chatglm_config = ChatGLMConfig(
        num_layers=2,
        vocab_size=20000,
        use_cache=True,
        multi_query_attention=True,
        multi_query_group_num=2,
        num_attention_heads=8,
        hidden_size=1024,
    )
    model = ChatGLMForConditionalGeneration(chatglm_config)

    engine = CaiInferEngine(
        tp_size=tp_size,
        pp_size=pp_size,
        model=model,
        max_output_len=max_output_len,
        micro_batch_size=micro_batch_size,
    )
    output = engine.generate(inputs)
    if dist.get_rank() == 0:
        assert len(output[0]) == max_output_len, f"{len(output)}, {max_output_len}"


@parameterize("tp_size", [1])
@parameterize("pp_size", [2])
@parameterize("max_output_len", [4])
@parameterize("micro_batch_size", [1])
@clear_cache_before_run()
def run_pipeline_inference_test(tp_size, pp_size, max_output_len, micro_batch_size):
    pipeline_inference_test(tp_size, pp_size, max_output_len, micro_batch_size)
    torch.cuda.empty_cache()


@parameterize("tp_size", [2])
@parameterize("pp_size", [2])
@parameterize("max_output_len", [4])
@parameterize("micro_batch_size", [1])
@clear_cache_before_run()
def run_tp_pipeline_inference_test(tp_size, pp_size, max_output_len, micro_batch_size):
    pipeline_inference_test(tp_size, pp_size, max_output_len, micro_batch_size)
    torch.cuda.empty_cache()


@parameterize("tp_size", [2])
@parameterize("pp_size", [1])
@parameterize("max_output_len", [2])
@parameterize("micro_batch_size", [1])
@clear_cache_before_run()
def run_tp_inference_test(tp_size, pp_size, max_output_len, micro_batch_size):
    pipeline_inference_test(tp_size, pp_size, max_output_len, micro_batch_size)
    torch.cuda.empty_cache()


def check_tp_pipeline_inference(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_tp_pipeline_inference_test()


def check_single_inference(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_tp_inference_test()
    run_pipeline_inference_test()


@pytest.mark.skipif(
    not CUDA_SUPPORT or not HAS_LIGHTLLM_KERNEL,
    reason="kv-cache manager engine requires cuda version to be higher than 11.5",
)
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_pipeline_inference():
    spawn(check_tp_pipeline_inference, nprocs=4)
    spawn(check_single_inference, nprocs=2)


if __name__ == "__main__":
    test_pipeline_inference()