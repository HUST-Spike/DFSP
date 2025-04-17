# logger_utils.py
import os
import logging
import datetime
import pprint

def setup_logger(config, mode="train", extra_info=None):
    """设置日志记录器
    
    Args:
        config: 配置参数对象 包含save_path、fusion、dataset等属性
        mode: 运行模式，可以是 "train" 或 "test"
        extra_info: 额外信息，将添加到日志文件名中

    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志文件名，包含实验参数
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    filename_parts = [mode, "log", config.fusion, config.dataset, config.clip_model]
    if mode == "train":
        filename_parts.append(str(config.train_batch_size))
        filename_parts.append(str(config.lr))
        filename_parts.append(str(config.context_length))
    if extra_info:
        filename_parts.append(str(extra_info))
    filename_parts.append(timestamp)

    # 使用与脚本同级的log文件夹
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{log_dir}/{('_').join(filename_parts)}.log"

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器，避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    #console_handler = logging.StreamHandler()
    #console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    #console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    #logger.addHandler(console_handler)
    
    # 记录初始配置信息
    logging.info(f"开始新的{mode}实验 - 配置参数:")
    logging.info(f"模式: {mode}")
    logging.info(f"数据集: {config.dataset}")
    logging.info(f"融合方式: {config.fusion}")
    if mode == "train":
        logging.info(f"批次大小: {config.train_batch_size}")
        logging.info(f"学习率: {config.lr}")
        logging.info(f"总训练轮次: {config.epochs}")
    logging.info(f"clip模型选择: {config.clip_model}")
    logging.info(pprint.pformat(vars(config)))
    
    return logger

def log_section(section_name):
    """记录一个新的日志区块标题
    
    Args:
        section_name: 区块名称
    """
    logging.info(f"\n{'='*20} {section_name} {'='*20}")
