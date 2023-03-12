from Quora_App.pipeline import Training_Pipeline
from Quora_App.logger import logging
import warnings
warnings.filterwarnings("ignore")

def main():
    try:
        train= Training_Pipeline()
        train.run_training_pipeline()
    except Exception as e:
        logging.error(e)

if __name__=="__main__":
    main()
