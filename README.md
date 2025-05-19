# PSI_PROJECT – Aplikacja AI z Flask i MobileNetV2

To jest prosta aplikacja webowa, która umożliwia wrzucenie zdjęcia i otrzymanie predykcji z modelu AI (MobileNetV2).

## ✅ Co robi ta aplikacja?

- Pozwala wgrać obrazek przez przeglądarkę
- Pokazuje wynik predykcji
- Zapisuje wszystkie predykcje do pliku `predictions.log`

---

## 🔧 Jak uruchomić projekt u siebie?

### 1. Sklonuj projekt z GitHub

Wejdź w terminal (np. w PyCharm albo systemowy) i wpisz:

```bash
git clone https://github.com/KrystianDrag24/PSI_PROJECT.git
cd PSI_PROJECT
```

2. Utwórz i aktywuj środowisko wirtualne (opcjonalnie, ale zalecane)
Windows:

```
python -m venv venv
venv\Scripts\activate
```


3. Zainstaluj wymagane biblioteki

```
pip install -r requirements.txt
```

4. Uruchom aplikację

```
python app.py
```

Otwórz przeglądarkę i przejdź na adres:

http://127.0.0.1:5000