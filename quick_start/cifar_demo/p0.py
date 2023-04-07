import colossalai
import os
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get('WORLD_SIZE', 2))
host = os.environ.get('MASTER_ADDR',"172.20.51.198")
port = int(os.environ.get('MASTER_PORT', 19090))

print("rank", rank)
print("local_rank", local_rank)
print("world_size", world_size)
print("host", host)
print("port", port)

# ./config.py refers to the config file we just created in step 1
colossalai.launch(config='./config.py',
           rank=rank,
           world_size=world_size,
           host=host,
           port=port,
           backend= 'nccl',
           local_rank=local_rank,
           seed=6,
           verbose=True)

from pathlib import Path
from colossalai.logging import get_dist_logger
import torch
import os
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34


# build logger
logger = get_dist_logger("engine")

# build resnet
model = resnet34(num_classes=10)

# build datasets
train_dataset = CIFAR10(
    root='./data',
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.2023, 0.1994, 0.2010]),
        ]
    )
)

test_dataset = CIFAR10(
    root='./data',
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.2023, 0.1994, 0.2010]),
        ]
    )
)

# build dataloaders
train_dataloader = get_dataloader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=gpc.config.BATCH_SIZE,
                                  num_workers=1,
                                  pin_memory=True,
                                  )

test_dataloader = get_dataloader(dataset=test_dataset,
                                 add_sampler=False,
                                 batch_size=gpc.config.BATCH_SIZE,
                                 num_workers=1,
                                 pin_memory=True,
                                 )

# build criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# lr_scheduler
lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)





engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                     optimizer,
                                                                     criterion,
                                                                     train_dataloader,
                                                                     test_dataloader,
                                                                     )


for epoch in range(gpc.config.NUM_EPOCHS):
    # execute a training iteration
    engine.train()
    for batch_id, (img, label) in enumerate(train_dataloader, 1):
        img = img.cuda()
        label = label.cuda()

        # set gradients to zero
        engine.zero_grad()

        # run forward pass
        output = engine(img)

        # compute loss value and run backward pass
        train_loss = engine.criterion(output, label)
        engine.backward(train_loss)

        # update parameters
        engine.step()
        print(f"{epoch+1}/{gpc.config.NUM_EPOCHS} - [{batch_id}/{len(train_dataloader)}] - train loss: {train_loss:.5}")


    # update learning rate
    lr_scheduler.step()

    # execute a testing iteration
    engine.eval()
    correct = 0
    total = 0
    for img, label in test_dataloader:
        img = img.cuda()
        label = label.cuda()

        # run prediction without back-propagation
        with torch.no_grad():
            output = engine(img)
            test_loss = engine.criterion(output, label)

        # compute the number of correct prediction
        pred = torch.argmax(output, dim=-1)
        correct += torch.sum(pred == label)
        total += img.size(0)

    logger.info(f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])

    
    from colossalai.utils import save_checkpoint, load_checkpoint  
    save_checkpoint('tmp_latest_engine.pt', epoch, model)