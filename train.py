#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.dfsp import DFSP
from parameters import parser, YML_PATH
from loss import loss_calu

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *

# for logging
import logging
import datetime
from logger_utils import setup_logger, log_section

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, logger):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
    train_losses = []

    log_section(logger, "训练初始化")
    logger.info(f"开始训练，总共 {config.epochs} 轮")

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):

            batch_img = batch[0].cuda()
            predict = model(batch_img, train_pairs)

            loss = loss_calu(predict, batch, config)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()
        scheduler.step()
        progress_bar.close()

        epoch_loss = np.mean(epoch_train_losses)
        progress_bar.write(f"epoch {i +1} train loss {epoch_loss}")
        logger.info(f"Epoch {i+1}/{config.epochs} - 训练损失: {epoch_loss:.6f}")
        train_losses.append(epoch_loss)

        if (i + 1) % config.save_every_n == 0:
            save_path = os.path.join(config.save_path, f"{config.fusion}_epoch_{i}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"模型保存至: {save_path}")

        print("Evaluating val dataset:")
        log_section(logger, f"Epoch {i+1} 验证集评估")
        logger.info("评估验证集...")
        loss_avg, val_result = evaluate(model, val_dataset, logger)
        print("Loss average on val dataset: {}".format(loss_avg))
        logger.info(f"验证集平均损失: {loss_avg:.6f}")

        if config.best_model_metric == "best_loss":
            if loss_avg.cpu().float() < best_loss:
                best_loss = loss_avg.cpu().float()
                print("Evaluating test dataset:")
                logger.info(f"发现新的最佳模型 (损失: {best_loss:.6f})，评估测试集...")
                evaluate(model, test_dataset, logger)
                best_model_path = os.path.join(config.save_path, f"{config.fusion}_best.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"最佳模型保存至: {best_model_path}")
        else:
            if val_result[config.best_model_metric] > best_metric:
                best_metric = val_result[config.best_model_metric]
                print("Evaluating test dataset:")
                logger.info(f"发现新的最佳模型 ({config.best_model_metric}: {best_metric:.4f})，评估测试集...")
                evaluate(model, test_dataset, logger)
                best_model_path = os.path.join(config.save_path, f"{config.fusion}_best.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"最佳模型保存至: {best_model_path}")

        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            logger.info("评估封闭世界下的测试集")
            model.load_state_dict(torch.load(os.path.join(
            config.save_path, f"{config.fusion}_best.pt"
        )))
            evaluate(model, test_dataset, logger)
    if config.save_model:
        final_model_path = os.path.join(config.save_path, f'final_model_{config.fusion}.pt')
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"最终模型保存至: {final_model_path}")


def evaluate(model, dataset, logger):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    result = ""
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in test_stats:
        if key in key_set:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)   
    logger.info(f"评估结果: {result}")  
    model.train()
    return loss_avg, test_stats



if __name__ == "__main__":
    config = parser.parse_args()
    load_args(YML_PATH[config.dataset], config)
    print(config)

    # 模型保存路径
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    path_components = [
        "train",
        config.fusion,
        config.dataset,
        config.clip_model,
        f"lr{config.lr}",
        f"bs{config.batch_size}",
        timestamp
    ]
    path_components = [comp for comp in path_components if comp]
    model_dir_name = "_".join(path_components)
    custom_save_dir = f"saved_models/{model_dir_name}"
    config.save_path = custom_save_dir

    # 声明logger
    logger = setup_logger(config, mode="train")
    logger.info("开始训练过程")
    logger.info(f"配置参数:\n{pprint.pformat(vars(config))}")

    # set the seed value
    set_seed(config.seed)
    logger.info(f"随机种子设置为: {config.seed}")

    dataset_path = config.dataset_path
    logger.info(f"数据集路径: {dataset_path}")

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural')

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    logger.info(f"训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}，测试集大小: {len(test_dataset)}")

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)
    logger.info(f"属性数量: {len(attributes)}，类别数量: {len(classes)}")

    model = DFSP(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    logger.info(f"优化器: Adam 学习率: {config.lr}，权重衰减: {config.weight_decay}")

    # if config.load_model is not False:
    #     model.load_state_dict(torch.load(config.load_model))

    os.makedirs(config.save_path, exist_ok=True)
    logger.info(f"模型将保存到: {config.save_path}")


    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, logger)

    # 更新最近模型保存路径， 便于test读取
    os.makedirs("saved_models", exist_ok=True)
    with open(f"saved_models/{config.dataset}_latest_model.txt", "w") as f: 
        f.write(os.path.join(config.save_path, f"{config.fusion}_best.pt"))
        logger.info(f"已更新最新模型路径记录")
        
    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    print("done!")
    logger.info("训练完成！配置已保存。")
