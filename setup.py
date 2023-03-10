import setuptools

with open("README.md","r",encoding = "utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "Quora-Question-Similarity-Classification"
AUTHOR_NAME = "Shivansh Kaushal"
AUTHOR_USER_NAME = "shivanshka"
SRC_REPO = "Quora_App"
AUTHOR_EMAIL = "kaushal.shivansh630@gmail.com"

setuptools.setup(
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
    packages=setuptools.find_packages(where='src') 
)