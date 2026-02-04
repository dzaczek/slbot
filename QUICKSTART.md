# Quick Start Guide

## 🚀 Szybki Start

### 1. Instalacja (Jednorazowo)
```bash
pip install -r requirements.txt
```

### 2. Test Systemu (Opcjonalne ale Zalecane)
```bash
python quick_test.py
```
Uruchomi bota na 30 sekund żeby sprawdzić czy wszystko działa.

### 3. Trening

**Opcja A: Interaktywny restart (zalecane)**
```bash
./restart_training.sh
```
- Wybierz czy kontynuować czy zacząć od nowa
- Automatycznie tworzy backupy

**Opcja B: Bezpośredni start**
```bash
python training_manager.py
```

### 4. Oglądanie Najlepszego Bota
```bash
python play_best.py
```

### 5. Analiza Postępów
```bash
python analyze_training.py
```

---

## 📋 Wszystkie Dostępne Komendy

### Trening
| Komenda | Opis |
|---------|------|
| `./restart_training.sh` | Interaktywny restart z opcją backup |
| `python training_manager.py` | Uruchom/wznów trening |
| `python quick_test.py` | Test 30s (weryfikacja systemu) |

### Granie
| Komenda | Opis |
|---------|------|
| `python play_best.py` | Graj najlepszym genomem |
| `python play_best.py neat-checkpoint-50` | Graj genomem z checkpointu |

### Analiza
| Komenda | Opis |
|---------|------|
| `python analyze_training.py` | Pokaż statystyki treningu |
| `tail -f training_log.txt` | Śledź logi na żywo |
| `tail -50 training_stats.csv` | Ostatnie 50 wyników |

---

## 🎯 Czego Się Spodziewać

### Generacje 1-10
- Bot będzie głupi, umrze szybko
- Niektóre zaczną zjadać 1-3 jedzenia
- Fitness: ~160 → ~400

### Generacje 10-30
- Bot zaczyna regularnie jeść
- 5-15 jedzenia na życie
- Fitness: ~400 → ~1000

### Generacje 30-50
- Bot unika ścian i wrogów
- 15-30 jedzenia
- Fitness: ~1000 → ~2000+

### Generacje 50+
- Bot jest mądry!
- Długie życie, dużo jedzenia
- Fitness: 2000+

---

## 🔧 Troubleshooting

### Bot nie je jedzenia po 50 generacjach?
```bash
# Zwiększ nagrodę w training_manager.py, linia ~214:
fitness_score += (diff * 200.0)  # Było 150.0
```

### Bot uderza w ściany?
```bash
# W spatial_awareness.py, linia ~209, zwiększ boost:
wall_danger = min(wall_danger * 2.0, 1.0)  # Było 1.5
```

### Trening jest za wolny?
```bash
# W training_manager.py, linia ~442, zmień:
NUM_WORKERS = 3  # Zmniejsz jeśli komputer jest wolny
```

### Chrome nie otwiera się?
```bash
# Sprawdź czy Chrome jest zainstalowany
# Zainstaluj webdriver-manager:
pip install webdriver-manager
```

---

## 📊 Parametry do Tuningu

Otwórz `training_manager.py` i znajdź te wartości:

```python
# Linia ~214 - Nagroda za jedzenie
fitness_score += (diff * 150.0)  # Zwiększ = bardziej agresywne jedzenie

# Linia ~218 - Timeout starvation
if time.time() - last_eat_time > 60:  # Zwiększ = więcej czasu na znalezienie jedzenia

# Linia ~266 - Nagroda za długość
fitness_score += (max_len * 20)  # Zwiększ = większa motywacja do wzrostu

# Linia ~263 - Waga survival time
fitness_score += (survival_time * 2.0)  # Zmniejsz = mniej pasywnego przeżycia
```

---

## 🎓 Pro Tips

1. **Start Od Nowa Po Zmianach**: Stare genomy są "utrwalone" w złych nawykach
2. **Monitoruj Logi**: `tail -f training_log.txt` pokaże co się dzieje
3. **Backup Często**: Najlepsze genomy mogą być nadpisane
4. **Patience**: Dobry bot potrzebuje 50-100+ generacji
5. **Eksperymentuj**: Zmień parametry i zobacz co działa!

---

## 📁 Ważne Pliki

| Plik | Opis |
|------|------|
| `training_manager.py` | Główna pętla treningu + fitness |
| `config_neat.txt` | Parametry NEAT (mutacje, populacja) |
| `spatial_awareness.py` | Przetwarzanie danych z gry |
| `ai_brain.py` | Wrapper dla sieci neuronowej |
| `browser_engine.py` | Kontrola przeglądarki |
| `best_genome.pkl` | Najlepszy wytrenowany genom |
| `neat-checkpoint-X` | Checkpointy (auto-save) |
| `training_stats.csv` | Historia wszystkich evaluacji |

---

## ❓ FAQ

**Q: Jak długo trwa trening?**  
A: 50 generacji × 50 genomów = 2500 evaluacji. Z 5 workerami ~1-2 godziny.

**Q: Czy mogę przerwać i wznowić?**  
A: Tak! Ctrl+C i potem `python training_manager.py` wznowi od ostatniego checkpointu.

**Q: Jak zapisać najlepszego bota?**  
A: Automatycznie zapisywany jako `best_genome.pkl` na końcu treningu.

**Q: Mogę trenować bez okna przeglądarki?**  
A: Tak, w `training_manager.py` ustaw `HEADLESS = True` (linia ~443).

**Q: Bot jest zbyt defensywny/agresywny?**  
A: Zmień balance między survival (linia 263) a food rewards (linia 214).

---

Powodzenia! 🐍🎮
