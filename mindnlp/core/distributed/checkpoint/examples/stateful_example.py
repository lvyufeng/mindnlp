# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

# pyre-unsafe


import os
import shutil

from mindnlp import core
from mindnlp import core.distributed as dist
from mindnlp import core.distributed.checkpoint as dcp
from mindnlp import core.multiprocessing as mp
from mindnlp import core.nn as nn
from core.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from core.distributed.device_mesh import init_device_mesh
from core.distributed.fsdp import FullyShardedDataParallel as FSDP


CHECKPOINT_DIR = f"~/{os.environ['LOGNAME']}/checkpoint"


class Model(core.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        core.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return core.rand(8, 8, device="cuda")


def _make_stateful(model, optim):
    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)


def _train(model, optim, train_steps=1):
    core.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


def _init_model(device, world_size):
    device_mesh = init_device_mesh(device, (world_size,))
    model = Model().cuda()
    model = FSDP(
        model,
        device_mesh=device_mesh,
        use_orig_params=True,
    )
    optim = core.optim.Adam(model.parameters(), lr=0.1)
    _make_stateful(model, optim)

    return model, optim


def run(rank, world_size, device="cuda"):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    core.cuda.set_device(rank)

    model, optim = _init_model(device, world_size)
    _train(model, optim, train_steps=2)

    dcp.save(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )

    # presumably do something else
    model, optim = _init_model(device, world_size)
    dcp.load(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )
    _train(model, optim, train_steps=2)


if __name__ == "__main__":
    world_size = core.cuda.device_count()
    print(f"Running stateful checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
