from setuptools import PackageFinder, find_packages, setup
from typing import List

with open("README.md","r",encoding = "utf-8") as f:
    long_description = f.read()

__version__ = "0.0.2"

REPO_NAME = "Quora-Question-Similarity-Classification"
AUTHOR_NAME = "Shivansh Kaushal"
AUTHOR_USER_NAME = "shivanshka"
SRC_REPO = "Quora_App"
AUTHOR_EMAIL = "kaushal.shivansh630@gmail.com"
REQUIREMENT_FILE_NAME = "requirements.txt"

def get_requirements_list()->List[str]:
    """
    Description: This function is going to return list of requirement mentioned in requirements.txt
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        return requirement_file.readlines().remove("-e .")

setup(
    name = SRC_REPO,
    version= __version__,
    author = AUTHOR_NAME,
    author_email= AUTHOR_EMAIL,
    description= "This application helps to detect similar questions in Quora",
    long_description=long_description,
    long_description_content = "text/markdown",
    url= f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}.git",
    project_urls = {
        "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir= {"":"src"},
    packages=find_packages(where='src') ,
    install_requires = get_requirements_list()
)