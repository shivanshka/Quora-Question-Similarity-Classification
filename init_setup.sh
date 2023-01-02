echo [$(date)]: "START"
echo [$(date)]: "creating environment"
conda create --prefix ./env python=3.8 -y
echo [$(date)]: "activate environment"
conda activate ./env
echo [$(date)]: "create folder and file structure"
python template.py
echo [$(date)]: "installing requirements"
pip install -r requirements.txt -q
echo [$(date)]: "END"