import logging
def my_log():
    # 创建Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建Handler

    # 文件Handler
    fileHandler = logging.FileHandler('log.log', mode='w', encoding='UTF-8')
    streamHandler = logging.StreamHandler()
    fileHandler.setLevel(logging.INFO)
    streamHandler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    # 添加到Logger中
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger