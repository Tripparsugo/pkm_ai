import random
import time

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping
from anchor import TMP_DIR
from lightning.pytorch.callbacks import RichProgressBar


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(1, 10)
        self.f1 = torch.nn.ELU(inplace=False)
        self.linear2 = torch.nn.Linear(10, 1)
        self.f2 = torch.nn.ELU(inplace=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.f1(x)
        x = self.linear2(x)
        x = self.f2(x)
        return x

    def training_step(self, train_batch, batch_idx):
        # train_batch = (torch.rand(64, 324), torch.rand(64, 22))
        mse_l = torch.nn.MSELoss()
        x, y = train_batch
        logits = self.forward(x)
        loss = mse_l(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        mse_l = torch.nn.MSELoss()
        x, y = val_batch
        logits = self.forward(x)
        loss = mse_l(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def f(x):
    return 1 + x[0] * 2 + x[0] * x[0]


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

    size = 100000
    xs = [[random.random() * 10] for x in range(0, size)]
    ys = [[f(x)] for x in xs]
    train_split = int(size * 0.8)
    validation_split = int(size * 0.9)

    data = list(zip([torch.tensor(x) for x in xs], [torch.tensor(y) for y in ys]))

    train_data = data[:train_split]
    validation_data = data[train_split:validation_split]
    test_data = data[validation_split:]

    train_loader = DataLoader(train_data, batch_size=1000, num_workers=10)
    validation_loader = DataLoader(validation_data, batch_size=1000, num_workers=10)

    # train
    logger = TensorBoardLogger(save_dir=f"{TMP_DIR}", name="test")
    trainer = pl.Trainer(max_epochs=5,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3),
                                    TQDMProgressBar(refresh_rate=5)
                                    ],
                         logger=logger,
                         log_every_n_steps=10,
                         profiler='simple'
                         )

    model = Model()
    loss_fn = torch.nn.MSELoss()
    xs_test = torch.stack(list(list(zip(*test_data))[0]))
    ys_test = torch.stack(list(list(zip(*test_data))[1]))

    pred_bt = model(xs_test)
    loss_bt = loss_fn(pred_bt, ys_test)
    start_time = time.time()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    end_time = time.time()
    pred_at = model(xs_test)

    loss_at = loss_fn(pred_at, ys_test)

    print(f"loss before training = {loss_bt}")
    print(f"loss after training = {loss_at}")
    print(f"training took = {end_time -start_time}s")

