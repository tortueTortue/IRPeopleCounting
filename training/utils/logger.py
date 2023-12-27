import logging
from datetime import date

def start_training_logging(model_name: str):
    logging.basicConfig(filename=f'logs/{model_name}_training_{date.today()}.log', level=logging.INFO)
    logging.info(f'Start training for model {model_name}')

    return logging

if __name__ == '__main__':

    l = start_training('model1')

    for i in range(10):
        l.info("new line")