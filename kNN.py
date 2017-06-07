import numpy as np
import random
import os, os.path, shutil
from sklearn.datasets import load_files  # Ładowanie metody load_files, zachowującej strukturę katalogu   
from sklearn.feature_extraction.text import TfidfVectorizer # W celu utworzenia wektorowej reprezentacji tesktu
from sklearn.neighbors import KNeighborsClassifier # W celu zastosowanie metody k najbliższych sąsiadów
from sklearn.linear_model import LogisticRegression # W celu utworzenia klasyfikatora, poprzez zastosowanie modelu liniowego.
from sklearn.model_selection import GridSearchCV # W celu regularyzacji regresji logistycznej
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # W celu usunięcia najczęściej spotykanych słów angielskich
from sklearn.metrics import accuracy_score, f1_score # Do pomiaru dokładności predykcji


class cd:
    """Menedżer kontekstowy do zmiany katalogu. 
      
           Klasa pobrana z Internetu. Żródło: StackOverflow.
           Bezpośrednie użycie metody os.chdir() bez tego rodzaju opakowania
           nie jest bezpiecznie.
   """
    def __init__(self, ścieżka):
        self.ścieżka = os.path.expanduser(ścieżka)

    def __enter__(self):
        self.inna_ścieżka = os.getcwd()
        os.chdir(self.ścieżka)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.inna_ścieżka)



class PodziałDanych:
    def dzielenie_danych(folder):
        """Metoda służy do podziału danych na treningowe i uczące

               Otwiera ona katalog 'folder' czyta jego podkatalogi, następnie tworzy dodatkowe podkatalogi
               o nazwach, które są nazwami dotychczasowych podkatalogów z dodatkiem '_train'
               oraz '_test', do których skopiuje losowe wybrane pliki z dotychczasowy katalogów. os
               Struktura katalogu 'folder' będzie zatem taka:
               - folder
               -- (dotychczasowe podkatalogi)
               -- treningowe
               --- (zawiera nazwy dotychczasowych katalogów z 80% plików)
               -- testowe
               --- (zawiera nazwy dotychczasowych katalogów z 20% plików)
        """
        pliki = os.listdir(folder)
        licznik=0
        if "treningowe" in pliki:
            shutil.rmtree(folder + "/treningowe")
            shutil.rmtree(folder + "/testowe")
        pliki = os.listdir(folder)
        if "treningowe" in pliki:
           pliki.remove("treningowe")
        if "testowe" in pliki:
           pliki.remove("testowe")
        print(pliki)
        nowy_folder = folder + "/treningowe"
        if not os.path.exists(nowy_folder):
            os.makedirs(nowy_folder)
        nowy_folder = folder + "/testowe"
        if not os.path.exists(nowy_folder):
            os.makedirs(nowy_folder)
        tren = folder + "/treningowe"
        with cd(tren):
           for nazwa_podkatalogu in pliki:
               if not os.path.exists(nazwa_podkatalogu):
                   os.makedirs(nazwa_podkatalogu)
        test = folder + "/testowe"
        with cd(test):
           for nazwa_podkatalogu in pliki:
               if not os.path.exists(nazwa_podkatalogu):
                   os.makedirs(nazwa_podkatalogu)
        with cd(folder):
            for nazwa_katalogu in pliki:
                katalog = nazwa_katalogu
                with cd(katalog):
                    f = []
                    
                    licznik = licznik+1
                    print("Wykonano podział danych dla folderu nr {}".format(licznik))
                    g = [nazwa for nazwa in os.listdir('.') if os.path.isfile(nazwa)]
                    leng = len(g)
                    print("Ilosc danych testowych:  ", leng//5) 
                    ciag_wylosowany = random.sample(g, leng//5)
                    g2 = ciag_wylosowany
                    g1 = [nazwa for nazwa in g if nazwa not in g2]
                    print("Ilosc danych treningowych:  {}".format(len(g1)))
                    for nazwa_pliku in g1:
                        shutil.copy2(nazwa_pliku, "C:/kNN/" + 
                               folder + "/treningowe/" + katalog + "/" + nazwa_pliku)
                    for nazwa_pliku in g2:
                        shutil.copy2(nazwa_pliku, "C:/kNN/" +
                               folder + "/testowe/" + katalog + "/" + nazwa_pliku)
        print("Koniec podziału na dane treningowe i testowe")

    def ładowanie_danych(folder):
       """Metoda pobierające dane z podfolderów.
           
             Zadaniem tej metody jest załadowanie danych z podfolderów 'treningowe'
             i 'testowe' za pomocą metody load_files z modułu sklearn_datasets.
       """
       tren = "/treningowe"
       test = "/testowe"
       dane_treningowe = load_files(folder + tren)
       dane_testowe = load_files(folder + test)
       return dane_treningowe, dane_testowe
                    


    def wektorowanie(folder, metoda="kNN", min = 5, max = 0.5, słowa_do_usunięcia = False):
        """Tworzenie macierzowej reprezentacji tekstów za pomocą klasy 'CountVectorizer' i jej metod 'fit' i 'transform'
               Za pomocą klasy 'CountVectorizer' i jej metod 'fit' i 'transform' tworzona jest macierzowa 
               reprezentacja tekstu. Reprezentacja ta stosuje rzadke macierze, zdefiniowane w pakiecie SciPy
        """
        print("Początek wektorowania....")
        PodziałDanych.dzielenie_danych(folder)
        dane = PodziałDanych.ładowanie_danych(folder)
        dane_treningowe, dane_testowe = dane[0], dane[1]
        teksty_uczace, etykiety1 = dane_treningowe.data, dane_treningowe.target
        teksty_testowe, etykiety2 = dane_testowe.data, dane_testowe.target
        
        if metoda == "kNN":
            wektoryzator = TfidfVectorizer(sublinear_tf=True, min_df=min, max_df=max, stop_words='english')
            X_treningowe = wektoryzator.fit_transform(teksty_uczace)
            print("Utworzono X_treningowe knn.")
            X_testowe = wektoryzator.transform(teksty_testowe)
            print("Utworzono X_testowe knn.")
        y_treningowe = etykiety1
        y_testowe = etykiety2
        print("Koniec wektorowania")
        return X_treningowe, y_treningowe, X_testowe, y_testowe, metoda 




class KlasyfikatorTekstów():
    """Klasa, służąca do klasyfikacji danych tekstowych z pomocą regresji logistycznej oraz metody k najbliższych sąsiadów.
        
           Używając modułów, dostępnych w pakiecie scikit-learn, metody klasy KlasyfikatorTekstów dokonują
           klasyfikacji danych tekstowych, zawartych w katalogu 'folder'. Zakłada się, że katalog 'folder'
           zawiera podkatalogi, których nazwy są zarazem etykietami kategorii, do których klasyfikowane
           będą dane testowe.
    """
    def __init__(self, folder, metoda):
        """Inicjalizator (pseudo-konstruktor) pobiera dane z katalogu 'folder' organizując je na potrzeby innych metod"""

        Xym = PodziałDanych.wektorowanie(folder, metoda)
        self.X_tren, self.y_tren, self.X_test, self.y_test = Xym[0], Xym[1], Xym[2], Xym[3]  
        self.metoda = Xym[4]    
   
        # Wyświetlane podstawowych informacji na temat typów danych, podlegających klasyfikacji.
        # kształtów ('shape') tablic oraz wielkości katalogów z danymi treningowymi i testowymi.

        print("Typ danych X_tren {} ".format(type(self.X_tren)))
        print("Kształt tablicy X_tren {} ".format(self.X_tren.shape))
        print("Typ danych X_test {} ".format(type(self.X_test)))
        print("Kształt tablicy X_test {} ".format(self.X_test.shape))
        print("Typ danych y_tren {} ".format(type(self.y_tren)))
        print("Kształt tablicy y_tren {} ".format(self.y_tren.shape))
        print("Typ danych y_test {} ".format(type(self.y_test)))
        print("Kształt tablicy y_testowe {} ".format(self.y_test.shape))
        print("Liczba plików w poszczególnych podfolderach: {} ".format(np.bincount(self.y_tren)))
        print("Liczba plików w poszczególnych podfolderach: {} ".format(np.bincount(self.y_test)))
    
        
    def regularyzacja_klasyfikatora(self):
 
        parametryC = {'n_neighbors': [5,10,15,20,25]}
        siatka = GridSearchCV(KNeighborsClassifier(), parametryC, cv = 5)
        y_treningowe = self.X_tren
        y_testowe = self.y_tren
        siatka.fit(y_treningowe, y_testowe)
        najlepszy_wynik = siatka.best_score_
        najlepsze_parametry = siatka.best_params_
        najlepsze_parametry = najlepsze_parametry["n_neighbors"]
        print("Najlepszy wynik po regularyzacji to: {}".format(siatka.best_score_))
        print("Najlepsze parametry po regularyzacji to: {}".format(najlepsze_parametry))
        
            #        wynik_dla_danych_testowych = siatka.score(X_testowe, y_testowe)
        return najlepszy_wynik, najlepsze_parametry


    def tworzenie_klasyfikatora(self, liczba_sąsiadów = 1):
        """Tworzenie klasyfikatora za pomocą modelu liniowego regresji logistycznej i ocena jego dokładności"""
        print("Początek tworzenia klasyfikatora; tworzenie_klasyfikatora: wywołanie metody wektorowanie...")
 
        print("Liczba sąsiadow to: {}".format(liczba_sąsiadów))
        knn = KNeighborsClassifier(liczba_sąsiadów)
        knn.fit(self.X_tren, self.y_tren)
        
        przewidywane = knn.predict(self.X_test)
        print("tworzenie_klasyfikatora(kNN)")
        wynik1 = accuracy_score(self.y_test, przewidywane)
        print("Wynik Accuracy na tekstach testowych: {} ".format(wynik1))
        wynik = wynik1
        print("Koniec tworzenia klasyfikatora, {} ".format(wynik))
        return wynik

    

    def wyświetlanie_informacji(self):
        """ Wyświetlanie informacji na temat tekstów, podlegających klasyfikacji i ich reprezentacji macierzowej """
                
        # Nie od rzeczy będzie wyświetlenie informacji na temat typu i wielkości zmiennej 'macierzowa_reprezentacja_tekstu'
        print()
        wek = self.wektorowanie()[2]
        nazwy_cech = wek.get_feature_names()
        print("Liczba cech: {}".format(len(nazwy_cech)))
        print("Pierwszych 10 cech: {}".format(nazwy_cech[:10]))

        # Tutaj wyświetlimy średnią dokładność, osiąganą w modelu liniowym poprzez użycie regresji logistycznej
        rezultat = self.tworzenie_klasyfikatora()
        print("Średnia dokładność modelu: {:3f}".format(np.mean(wynik)))

        # Tutaj pokazujemy, czy regularyzacja poprawia wyniki.
        rezultat = self.regularyzacja_klasyfikatora()
        print ("Średnia dokładność po regularyzacji: {:3f}".format(wynik_grid[0]))
        print ("Najlepszy parametr regularyzacji: ", wynik_grid[1])

        
def main(): 
    print("main: tworzenie obiektu KlasyfikatorTekstów...")
    klasyfikator_tekstów = KlasyfikatorTekstów("text_files", "kNN")
    print("main: koniec konstruktora KlasyfikatorTekstów TRWA OBLICZANIE...")
    wynik_grid = klasyfikator_tekstów.regularyzacja_klasyfikatora()
    wynik = klasyfikator_tekstów.tworzenie_klasyfikatora(wynik_grid[1])
    

if __name__ == "__main__":
    main()
        

