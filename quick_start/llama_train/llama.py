import argparse
import os

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get('WORLD_SIZE', 2))
host = os.environ.get('MASTER_ADDR',"172.20.51.198")
port = int(os.environ.get('MASTER_PORT', "19090"))

os.environ["RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["MASTER_ADDR"] = str(host)
os.environ["MASTER_PORT"] = str(port)

print("rank", rank)
print("local_rank", local_rank)
print("world_size", world_size)
print("host", host)
print("port", port)


import loralib as lora
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, SFTDataset, SupervisedDataset
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMLM
from coati.models.gpt import GPTLM
from coati.models.llama import LlamaLM
from coati.models.opt import OPTLM
from coati.trainer import SFTTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ColoParameter


def train(args):
    # configure strategy
    stage = 3
    strategy = ColossalAIStrategy(stage=stage, placement_policy='cpu')


    # configure model
    with strategy.model_init_context():
        model = LlamaLM(pretrained=args.pretrain, lora_rank=args.lora_rank, checkpoint=True)
        model = model.to(torch.float16).to(torch.cuda.current_device())

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.eos_token = '<\s>'
    
    tokenizer.pad_token = tokenizer.eos_token
    max_len = args.max_len

    tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, model)

    if stage==3:
        # this is a hack to deal with the resized embedding
        # to make sure all parameters are ColoParameter for Colossal-AI Gemini Compatiblity
        for name, param in model.named_parameters():
            if not isinstance(param, ColoParameter):
                sub_module_name = '.'.join(name.split('.')[:-1])
                weight_name = name.split('.')[-1]
                sub_module = model.get_submodule(sub_module_name)
                setattr(sub_module, weight_name, ColoParameter(param))

    # configure optimizer
    optim = HybridAdam(model.parameters(), lr=args.lr, clipping_norm=1.0)

    logger = get_dist_logger()

    # configure dataset
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                        data_path=args.dataset,
                                        max_datasets_size=args.max_datasets_size,
                                        max_length=max_len)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=True,
                                           seed=42,
                                           drop_last=True,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())
    else:
        train_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=data_collator,
                                  pin_memory=True)


    trainer = SFTTrainer(model=model,
                         strategy=strategy,
                         optim=optim,
                         train_dataloader=train_dataloader,
                         eval_dataloader=None,
                         batch_size=args.batch_size,
                         max_epochs=args.max_epochs,
                         accimulation_steps=args.accimulation_steps)

    trainer.fit(logger=logger, log_interval=args.log_interval)

    # save model checkpoint after fitting on only rank0
    trainer.save_model(path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2', 'colossalai_zero2_cpu'],
                        default='colossalai_gemini')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='llama')
    parser.add_argument('--pretrain', type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument('--dataset', type=str, default="datasets/instinwild_ch.json")
    parser.add_argument('--max_datasets_size', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='model_saved_here/tmp')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--log_interval', type=int, default=10, help="how many steps to log")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--accimulation_steps', type=int, default=1)
    args = parser.parse_args()
    train(args)
