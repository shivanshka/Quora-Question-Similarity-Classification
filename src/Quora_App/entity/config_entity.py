from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_dir: Path
    download_url: str
    zipped_download_dir: Path
    download_file_name: str

@dataclass(frozen=True)
class DataValidationConfig:
    schema_file_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    transformed_train_dir: Path
    transformed_test_dir: Path
    feature_eng_object_file_path: Path
    preprocessed_object_file_path: Path
    word2tfidf_object_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path: Path
    base_accuracy: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_evaluation_file_path: Path
    time_stamp: str

@dataclass(frozen=True)
class ModelPusherConfig:
    export_dir_path: Path

@dataclass(frozen=True)
class TrainingPipelineConfig:
    artifact_dir: Path
    
