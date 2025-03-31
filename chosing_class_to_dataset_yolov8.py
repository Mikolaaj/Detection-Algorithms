import os
import tkinter as tk
from tkinter import filedialog

def wybierz_katalog():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Wybierz katalog z plikami YOLO")
    return folder_path

def zmien_klase_w_plikach(folder, nowa_klasa):
    if not os.path.isdir(folder):
        print("Podany katalog nie istnieje.")
        return
    
    for plik in os.listdir(folder):
        if plik.endswith(".txt"):  # Przetwarzamy tylko pliki tekstowe
            sciezka_pliku = os.path.join(folder, plik)
            with open(sciezka_pliku, "r") as f:
                linie = f.readlines()
            
            nowe_linie = []
            for linia in linie:
                dane = linia.strip().split()
                if dane:  # Sprawdzamy, czy linia nie jest pusta
                    dane[0] = str(nowa_klasa)  # Zamiana numeru klasy
                    nowe_linie.append(" ".join(dane))
            
            with open(sciezka_pliku, "w") as f:
                f.write("\n".join(nowe_linie))
    
    print(f"✅ Zmieniono wszystkie klasy na {nowa_klasa} w plikach w katalogu: {folder}")


if __name__ == "__main__":
    katalogi = []
    for i in range(3):  # Pozwala wybrać 3 katalogi na początku
        print(f"Wybierz katalog {i+1}:")
        katalog = wybierz_katalog()
        if katalog:
            katalogi.append(katalog)
        else:
            print("❌ Nie wybrano katalogu. Pomijam.")
    
    for i, katalog in enumerate(katalogi):
        try:
            nowa_klasa = int(input(f"Podaj nową wartość klasy dla katalogu {i+1}: "))
            zmien_klase_w_plikach(katalog, nowa_klasa)
        except ValueError:
            print("❌ Podano niepoprawną liczbę. Pomijam katalog.")
