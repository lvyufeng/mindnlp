# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

import os
import shutil
import traceback

from mindnlp import core
from mindnlp import core.distributed as dist
from mindnlp import core.distributed.checkpoint as dcp
from mindnlp import core.multiprocessing as mp
from mindnlp import core.nn as nn
from mindnlp import core.nn.functional as F
from core.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from core.distributed.fsdp import FullyShardedDataParallel as FSDP
from core.distributed.tensor.device_mesh import init_device_mesh


DEVICE = "cuda"
NUM_EPOCHS = 1000
SAVE_PERIOD = 10
FAULT_PERIOD = 25
CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"


class InjectedException(Exception):
    pass


class Model(core.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(8, 32)
        self.net2 = nn.Linear(32, 128)
        self.net3 = nn.Linear(128, 64)
        self.net4 = nn.Linear(64, 8)
        self.net5 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        x = F.sigmoid(self.net5(x))
        return x


def _init_model(rank, world_size):
    device_mesh = init_device_mesh(DEVICE, (world_size,))

    # Create a dummy model and wrap it in FSDP
    model = Model().cuda()
    device_mesh = init_device_mesh(DEVICE, (world_size,))
    model = FSDP(model, device_mesh=device_mesh, use_orig_params=True)

    optim = core.optim.Adam(model.parameters(), lr=0.0001)

    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)

    return model, optim


def _print(msg):
    if dist.get_rank() == 0:
        print(msg)


def _input():
    x = core.rand(128, 8, device="cuda")
    y = core.zeros(128, 1, device="cuda")

    y[core.sum(x, dim=1) >= 4] = 1.0

    return x, y


def run(rank, world_size):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    core.cuda.set_device(rank)

    model, optim = _init_model(rank, world_size)
    state_dict = {"model": model, "optim": optim}
    loss_calc = core.nn.BCELoss()

    f = None
    for epoch in range(NUM_EPOCHS):
        try:
            core.manual_seed(epoch)
            x, y = _input()

            loss = loss_calc(model(x), y)

            _print(f"{epoch=} {loss=}")

            loss.backward()
            optim.step()
            optim.zero_grad()

            if epoch % SAVE_PERIOD == 0:
                if f is not None:
                    f.result()
                f = dcp.state_dict_saver.async_save(
                    state_dict, checkpoint_id=CHECKPOINT_DIR
                )

            if FAULT_PERIOD > 0 and epoch % FAULT_PERIOD == 0:
                raise InjectedException("Fault injection!")

        except InjectedException as e:
            dist.barrier()

            _print("Trainer encountered exception:")
            traceback.print_tb(e.__traceback__)

            _print("Reloading model from last checkpoint!")
            if f is not None:
                f.result()
            dcp.load(state_dict)


if __name__ == "__main__":
    world_size = core.cuda.device_count()
    print(f"Running an example of Async Checkpointing on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
