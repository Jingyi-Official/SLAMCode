'''
Description: 
Author: 
Date: 2022-09-19 21:49:18
LastEditTime: 2023-03-27 03:10:03
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''
import os
import warnings

import hydra
from omegaconf import DictConfig,OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

import torch
from torch.utils.data import Dataset
from torch.nn import Module
import wandb

warnings.filterwarnings("ignore")

from typing import Any, Callable, Dict, List

# A logger for this file
import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config")
def main(cfg:DictConfig):
    log.info("Set device.")
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    log.info("Set seed for random number generators in pytorch, numpy and python.random")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating dataset <{cfg.dataset._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: Module = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer: Module = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = []  
    for _, cb in cfg.get("callback").items():
        if isinstance(cb, DictConfig) and "_target_" in cb:
            log.info(f"Instantiating callback <{cb._target_}>")
            callbacks.append(hydra.utils.instantiate(cb))
    
    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger: LightningLoggerBase = hydra.utils.instantiate(cfg.logger)

    log.info(f"Instantiating Evaluator <{cfg.evaluator._target_}>")
    evaluator: Module = hydra.utils.instantiate(cfg.evaluator, device = device, dataset = dataset)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, device = device, model=model, optimizer=optimizer, callbacks=callbacks, logger=logger, log=log, evaluator=evaluator)

    log.info("** Start **")  
    trainer.fit(dataset=dataset)

    log.info("** End **")
    logger.log_hyperparams(cfg)
    

if __name__ == "__main__":
    main()