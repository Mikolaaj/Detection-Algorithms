# GIT
git clone https://...

git status

git --version

git config --global user.name "nazwa uzytkownika"

git config --global user.email "email uzytkownika"

git add .

git commit -m "tekst"

git push

# Python
python3 --version

* Choosing python version
    sudo update-alternatives --config python3

* Instalation version of python
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 3
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

* Virutal Environment
    - Making: python3 -m venv <nazwa srodowiska>
    - Activating: source <nazwa>/bin/activate
    - Deactivating: deactivate

# Linux



# WSL

* Check
    - wsl --list --verbose
    - wsl --status
    - wsl --list
    - wsl --list --online
    - wsl -l -o

* Shut down
    - wsl --shutdown

* Delete
    - wsl --unregister <nazwa_dystrybucji>
    - wsl --unregister Ubuntu

* Install
    - wsl --install
    - wsl -d <nazwa_dystrybucji>
    - wsl -d Ubuntu
    - wsl --install -d Ubuntu-20.04

