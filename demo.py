from Quora_App.pipeline import Training_Pipeline2
from Quora_App.logger import logging
import warnings
warnings.filterwarnings("ignore")

def main():
    try:
        train= Training_Pipeline2()
        train.run_training_pipeline()
    except Exception as e:
        print(e)

if __name__=="__main__":
    main()
