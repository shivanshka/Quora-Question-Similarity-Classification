from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    is_ingested:bool
    message:str
    data_file_path: Path

@dataclass(frozen=True)
class DataValidationArtifact:
    schema_file_path: Path
    is_validated: bool
    message: str

@dataclass(frozen=True)
class DataTransformationArtifact:
    is_transformed: bool
    message: str
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    preprocessed_obj_file_path: Path
    feat_eng_obj_file_path: Path
    word2tfidf_object_file_path: Path

@dataclass(frozen=True)
class ModelTrainerArtifact:
    is_trained: bool
    message: str
    trained_model_file_path: Path
    logloss_value: float
    roc_auc_score: float
    threshold: float

@dataclass(frozen=True)
class ModelEvaluationArtifact:
    is_model_accepted: bool
    evaluated_model_path: Path

@dataclass(frozen=True)
class ModelEvaluationArtifact:
    is_model_pushed: bool
    export_model_file_path: Path