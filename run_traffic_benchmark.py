from typing import Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

from tsl import logger
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.engines import Predictor
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn import models
from tsl.utils.casting import torch_to_numpy


def get_model_class(model_str):
    # Spatiotemporal Models ###################################################
    if model_str == 'dcrnn':
        model = models.DCRNNModel  # (Li et al., ICLR 2018)
    elif model_str == 'gwnet':
        model = models.GraphWaveNetModel  # (Wu et al., IJCAI 2019)
    elif model_str == 'evolvegcn':
        model = models.EvolveGCNModel  # (Pereja et al., AAAI 2020)
    elif model_str == 'agcrn':
        model = models.AGCRNModel  # (Bai et al., NeurIPS 2020)
    elif model_str == 'grugcn':
        model = models.GRUGCNModel  # (Guo et al., ICML 2022)
    elif model_str == 'gatedgn':
        model = models.GatedGraphNetworkModel  # (Satorras et al., 2022)
    elif model_str == 'stcn':
        model = models.STCNModel
    elif model_str == 'transformer':
        model = models.TransformerModel
    elif model_str == 'nexusqn':
        model = models.NexuSQNModel
    # Temporal Models #########################################################
    elif model_str == 'ar':
        model = models.ARModel
    elif model_str == 'var':
        model = models.VARModel
    elif model_str == 'rnn':
        model = models.RNNModel
    elif model_str == 'fcrnn':
        model = models.FCRNNModel
    elif model_str == 'tcn':
        model = models.TCNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name):
    if dataset_name == 'la':
        dataset = MetrLA(impute_zeros=True)  # From Li et al. (ICLR 2018)
    elif dataset_name == 'bay':
        dataset = PemsBay()  # From Li et al. (ICLR 2018)
    elif dataset_name == 'pems3':
        dataset = PeMS03()  # From Guo et al. (2021)
    elif dataset_name == 'pems4':
        dataset = PeMS04()  # From Guo et al. (2021)
    elif dataset_name == 'pems7':
        dataset = PeMS07()  # From Guo et al. (2021)
    elif dataset_name == 'pems8':
        dataset = PeMS08()  # From Guo et al. (2021)
    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    return dataset


def get_logger(cfg: DictConfig, exp: Experiment) -> Optional[Logger]:
    if cfg.logger is None:
        return None
    assert 'backend' in cfg.logger, \
        "cfg.logger must have a 'backend' attribute."
    if cfg.logger.backend == 'wandb':
        exp_logger = WandbLogger(name=cfg.run.name,
                                 save_dir=cfg.run.dir,
                                 offline=cfg.logger.offline,
                                 project=cfg.logger.project,
                                 config=exp.get_config_dict(),
                                 tags=cfg.tags)
    elif cfg.logger.backend == 'neptune':
        exp_logger = NeptuneLogger(project_name=cfg.neptune.project,
                                   experiment_name=cfg.run.name,
                                   save_dir=cfg.run.dir,
                                   tags=cfg.tags,
                                   params=exp.get_config_dict(),
                                   debug=cfg.neptune.offline)
    elif cfg.logger.backend == 'tensorboard':
        exp_name = f'{cfg.run.name}_{"_".join(cfg.tags)}'
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=exp_name)
    else:
        raise ValueError(f"Logger {cfg.logger.backend} not available.")
    return exp_logger


def run_traffic(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset.name)

    # encode time of the day and use it as exogenous variable
    covariates = {'u': dataset.datetime_encoded('day').values}

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=dataset.mask,
        connectivity=adj,
        covariates=covariates,
        horizon=cfg.horizon,
        window=cfg.window,
        stride=cfg.stride,
    )

    transform = {'target': StandardScaler(axis=(0, 1))}  # axis: time&space

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon,
                        exog_size=torch_dataset.input_map.u.shape[-1])

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mape': torch_metrics.MaskedMAPE(),
        'mae_at_15': torch_metrics.MaskedMAE(at=2),  # 3rd is 15 min
        'mae_at_30': torch_metrics.MaskedMAE(at=5),  # 6th is 30 min
        'mae_at_60': torch_metrics.MaskedMAE(at=11),  # 12th is 1 h
    }

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    exp_logger = get_logger(cfg, exp)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=cfg.grad_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(predictor, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()
    trainer.test(predictor, datamodule=dm)

    output = trainer.predict(predictor, dataloaders=dm.test_dataloader())
    output = predictor.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('mask', None))
    res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
               test_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
               test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    output = trainer.predict(predictor, dataloaders=dm.val_dataloader())
    output = predictor.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('mask', None))
    res.update(
        dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
             val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
             val_mape=numpy_metrics.mape(y_hat, y_true, mask)))

    return res


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic,
                     config_path='Config/traffic',
                     config_name='default')
    res = exp.run()
    logger.info(res)
