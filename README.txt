""" The following is using python > 3.6.0 but < 3.8, check it:"""
python3 -m pip --version

""" Create a virtual environment called venv:"""
python3 -m venv venv

""" Activate the virtual environment for running scripts"""
source venv/bin/activate

""" Install the modules """
pip3 install -r requirements.txt

""" See list of installed modules """
pip3 freeze

""" Deactivate the environment """
deactivate

""" PyCharm settings """
PyCharm | Preferences | Editor | Code Style | Python -> check "Use tab character"
