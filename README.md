# 🎙️ Generator Skryptów VO/TTS

Prosta aplikacja webowa do generowania profesjonalnych skryptów dla lektorów, wykorzystująca OpenAI GPT-4.

## 🚀 Funkcje

- **Prosty formularz** - wybierz język i wpisz brief
- **Presety** - monolog / call center (agent-only) / dialogi (sceny)
- **Automatyczne generowanie** - AI tworzy wysokiej jakości skrypt dla lektorów
- **Post-processing** - zaawansowane czyszczenie i optymalizacja tekstu
- **Pobieranie plików** - eksport do .txt
- **Responsywny design** - działa na telefonie i komputerze
- **Team-only access** - opcjonalne hasło (APP_PASSWORD)

## 📋 Wymagania

- Python 3.11+
- Klucz API OpenAI
- Konto na Render.com (darmowy plan wystarczy)

## 🛠️ Instalacja lokalna

1. Sklonuj repozytorium lub skopiuj pliki do folderu projektu

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

3. Ustaw zmienną środowiskową z kluczem API:
```bash
export OPENAI_API_KEY='twój-klucz-api'
```

Opcjonalnie (team-only):
```bash
export APP_PASSWORD='haslo-dla-zespolu'
export APP_SECRET_KEY='dlugi-losowy-sekret'
```

Opcjonalnie (domyślny model):
```bash
export OPENAI_MODEL='gpt-4.1'
```

4. Uruchom aplikację:
```bash
python app.py
```

5. Otwórz w przeglądarce: `http://localhost:5000`

## 🌐 Deployment na Render.com

### Krok 1: Przygotowanie repozytorium

1. Stwórz nowe repozytorium na GitHub
2. Dodaj wszystkie pliki projektu:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/twoja-nazwa/script-generator.git
git push -u origin main
```

### Krok 2: Konfiguracja na Render.com

1. Zaloguj się na [render.com](https://render.com)
2. Kliknij **"New +"** → **"Web Service"**
3. Połącz swoje konto GitHub i wybierz repozytorium
4. Skonfiguruj:
   - **Name**: `script-generator` (lub inna nazwa)
   - **Region**: wybierz najbliższy
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app` (powinno się wypełnić automatycznie z Procfile)
   - **Instance Type**: `Free`

5. W sekcji **Environment Variables** dodaj:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: twój klucz API OpenAI

   Opcjonalnie (żeby tylko zespół miał dostęp):
   - **Key**: `APP_PASSWORD`
   - **Value**: hasło zespołu

   Rekomendowane przy `APP_PASSWORD`:
   - **Key**: `APP_SECRET_KEY`
   - **Value**: długi losowy sekret (do podpisywania ciasteczek)

   Opcjonalnie:
   - **Key**: `OPENAI_MODEL`
   - **Value**: np. `gpt-4.1` albo `gpt-4o`

6. Kliknij **"Create Web Service"**

### Krok 3: Czekaj na deployment

Render automatycznie:
- Zainstaluje zależności
- Uruchomi aplikację
- Przydzieli URL (np. `https://script-generator-xyz.onrender.com`)

⏱️ Pierwszy deployment zajmuje ~5-10 minut.

### Krok 4: Testowanie

Otwórz przydzielony URL i przetestuj aplikację!

## 📁 Struktura projektu

```
script-generator-web/
├── app.py              # Główna aplikacja Flask
├── requirements.txt    # Zależności Python
├── Procfile           # Konfiguracja dla Render
├── runtime.txt        # Wersja Python
├── templates/
│   └── index.html     # Template HTML
└── static/
    └── style.css      # Style CSS
```

## 🔑 Zdobycie klucza API OpenAI

1. Idź na [platform.openai.com](https://platform.openai.com)
2. Zaloguj się / zarejestruj
3. Idź do **API keys** → **Create new secret key**
4. Skopiuj klucz (nie będziesz go więcej widział!)
5. Dodaj środki na konto (API wymaga prepaid)

## ⚠️ Ważne uwagi

### Render.com Free Tier

- ✅ **Darmowy hosting** - brak opłat za hosting
- ⏱️ **Spin down** - serwis usypia po 15 min nieaktywności
- 🐌 **Cold start** - pierwsze żądanie po wznowieniu trwa ~30-60s
- 📊 **750h/miesiąc** - wystarczy dla małego zespołu

### Koszty OpenAI

- Model GPT-4 kosztuje około **$0.03 na 1K tokenów wejściowych** i **$0.06 na 1K tokenów wyjściowych**
- Każde wygenerowanie skryptu to ~5-8K tokenów = około **$0.30-0.50 za skrypt**
- Monitoruj użycie w panelu OpenAI

### Bezpieczeństwo

- **NIGDY** nie commituj klucza API do repozytorium
- Używaj zmiennych środowiskowych
- Dodaj `.env` do `.gitignore` jeśli używasz lokalnie
- Jeśli ustawisz `APP_PASSWORD`, aplikacja wymaga zalogowania hasłem

## 🎨 Dostosowywanie

### Zmiana języków

Edytuj `templates/index.html`, sekcja `<select id="language">`:

```html
<option value="Twój język (kod)">Nazwa języka</option>
```

### Zmiana modelu AI

W `app.py`, funkcja `generate_script()`, zmień:

```python
model="gpt-4o"  # możesz zmienić na "gpt-4o-mini" (tańszy) lub inny
```

### Dostosowanie systemu promptów

Edytuj sekcję `messages` w funkcji `generate_script()` w `app.py`.

## 🐛 Rozwiązywanie problemów

### Aplikacja nie startuje

- Sprawdź logi w Render Dashboard
- Upewnij się, że `OPENAI_API_KEY` jest ustawiony
- Zweryfikuj że wszystkie pliki są w repozytorium

### Błąd OpenAI

- Sprawdź czy klucz API jest poprawny
- Zweryfikuj czy masz środki na koncie OpenAI
- Sprawdź limit rate (quota) w panelu OpenAI

### Timeout

- Pierwsze żądanie po cold start trwa dłużej
- Generowanie skryptu zajmuje 20-40s - to normalne

## 📞 Wsparcie

Jeśli napotkasz problemy:
1. Sprawdź logi w Render Dashboard
2. Sprawdź status API OpenAI: [status.openai.com](https://status.openai.com)
3. Zweryfikuj konfigurację zmiennych środowiskowych

## 📝 Licencja

Ten projekt jest otwarty do użytku zgodnie z potrzebami Twojego zespołu.

---

**Enjoy! 🎉**


