import os
from pathlib import Path

package_name = "Quora_App"

list_of_files = [
    ".github/workflows/.gitkeep",
    ".github/workflows/ci_cd.yaml",
    "config/config.yaml",
    "config/schema.yaml",
    "dvc.yaml",
    "params.yaml",
     f"src/{package_name}/__init__.py",
    f"src/{package_name}/config/configuration.py",
    f"src/{package_name}/config/__init__.py",
    f"src/{package_name}/constants/__init__.py",
    f"src/{package_name}/entity/__init__.py",
    f"src/{package_name}/entity/config_entity.py",
    f"src/{package_name}/entity/artifact_entity.py",
    f"src/{package_name}/entity/model_factory.py",
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/utils/utils.py",
    f"src/{package_name}/exception/__init__.py",
    f"src/{package_name}/logger/__init__.py",
    f"src/{package_name}/exception/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_validation.py",
    f"src/{package_name}/components/data_transformation.py",
    f"src/{package_name}/components/model_trainer.py",
    f"src/{package_name}/components/model_evaluater.py",
    f"src/{package_name}/components/model_pusher.py",
    f"src/{package_name}/pipeline/__init__.py",
    f"src/{package_name}/pipeline/training_pipeline.py",
    f"src/{package_name}/pipeline/prediction_pipeline.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "tox.ini",
    "research/stage_0_template.ipynb",
    "main.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
            print(f"Creating empty file: {filepath}")
    
    else:
        print(f"File: {filename} already exists")