# Ulepszenia Bota - Co Zostało Zmienione

## Problem
Bot był głupi - umierał po 20 sekundach przez starvation (głodowanie), nie jadł jedzenia, tylko przeżywał początkowy czas.

## Wprowadzone Zmiany

### 1. **MASYWNIE Zwiększone Nagrody za Jedzenie**
- **Poprzednio**: 25 punktów za zjedzenie
- **Teraz**: **150 punktów za zjedzenie** (6x więcej!)
- **Dlaczego**: Bot musi wiedzieć, że jedzenie jest NAJWAŻNIEJSZE

### 2. **Zwiększony Timeout Starvation**
- **Poprzednio**: 20 sekund bez jedzenia = śmierć
- **Teraz**: **60 sekund** (3x więcej czasu)
- **Dlaczego**: Bot potrzebuje czasu żeby nauczyć się łapać jedzenie

### 3. **Zmniejszona Waga Survival Time**
- **Poprzednio**: 5 punktów za sekundę
- **Teraz**: **2 punkty za sekundę**
- **Dlaczego**: Nie chcemy żeby bot tylko przeżywał - chcemy żeby jadł!

### 4. **Zwiększona Nagroda za Długość**
- **Poprzednio**: 5 punktów za segment
- **Teraz**: **20 punktów za segment** (4x więcej!)
- **Dlaczego**: Długość = sukces = główny cel

### 5. **Penalty za Collision**
- **Nowe**: Jeśli bot umrze przez collision w <15s, fitness × 0.3
- **Dlaczego**: Odstraszanie od samobójczego zachowania

### 6. **Penalty za Starvation**
- **Nowe**: Fitness × 0.5 jeśli umrze przez starvation
- **Dlaczego**: Motywacja do jedzenia

### 7. **Incremental Food Reward**
- **Nowe**: +0.1 punktu za zbliżanie się do jedzenia
- **Dlaczego**: Pomaga botowi nauczyć się że powinien iść w stronę jedzenia

### 8. **Lepsza Detekcja Ścian**
- **Poprzednio**: Ściany wykrywane tylko w kierunku od centrum
- **Teraz**: Każdy sektor sprawdzany osobno
- **Boost**: Danger × 1.5 dla ścian
- **Dlaczego**: Bot musi wiedzieć gdzie są ściany w KAŻDYM kierunku

### 9. **Większa Populacja**
- **Poprzednio**: 30 genomów
- **Teraz**: **50 genomów**
- **Dlaczego**: Więcej różnorodności = szybsze uczenie

### 10. **Agresywniejsza Ewolucja**
- Zwiększone conn_add_prob: 0.6 → 0.7
- Zmniejszone node_add_prob: 0.3 → 0.2 (wolniejszy wzrost complexity)
- Zwiększone elitism: 2 → 3 (więcej najlepszych przeżywa)
- **Dlaczego**: Szybsze eksplorowanie, ale z kontrolą nad complexity

## Jak Wznowić Trening

### Opcja 1: Kontynuuj ze Starym Genomem (Powolne Uczenie)
```bash
python training_manager.py
# Automatycznie załaduje neat-checkpoint-100
```

**Problem**: Stare genomy są już "utrwalone" w złych nawykach

### Opcja 2: START OD NOWA (ZALECANE!)
```bash
# Backup starych checkpointów
mkdir old_training
mv neat-checkpoint-* old_training/
mv best_genome.pkl old_training/
mv training_stats.csv old_training/training_stats_old.csv

# Start fresh
python training_manager.py
```

**Zaleta**: Nowe genomy od razu uczą się z nowymi nagrodami!

### Opcja 3: Hybrydowa - Stwórz Nową Populację ale z Inspiracją
```bash
# Usuń checkpointy ale zostaw training_stats
rm neat-checkpoint-*
python training_manager.py
```

## Czego Się Spodziewać

### Pierwsze 10 Generacji:
- Bot nadal będzie umierał szybko (starvation/collision)
- Ale niektóre genomy zaczną zjadać 1-3 jedzenia
- Fitness powinna wzrosnąć z ~160 do ~400-600

### Generacje 20-50:
- Bot powinien regularnie zjadać 5-15 jedzenia
- Przeżycie 30-60 sekund
- Fitness 800-1500

### Generacje 50+:
- Bot powinien zjadać 20+ jedzenia
- Przeżycie >1 minuty
- Fitness >2000
- Unikanie ścian i innych węży

## Jak Sprawdzić Postęp

```bash
# Analiza statystyk
python analyze_training.py

# Oglądaj najlepszego bota
python play_best.py

# Check logi
tail -f training_log.txt
```

## Parametry do Dalszego Tuningu

Jeśli bot nadal nie je:
1. Zwiększ nagrodę za jedzenie do 200+
2. Zwiększ penalty za starvation (fitness × 0.2)
3. Dodaj bonus za zbliżanie się do jedzenia (+0.5)

Jeśli bot uderza w ściany:
1. Zwiększ wall danger boost: 1.5 → 2.0
2. Dodaj penalty za collision: fitness × 0.1

Jeśli bot je ale jest zbyt defensywny:
1. Zmniejsz body_proximity danger
2. Zwiększ food rewards jeszcze bardziej

## Debug Tips

```bash
# Zobacz ostatnie 50 wyników
tail -50 training_stats.csv

# Zlicz przyczyny śmierci
awk -F',' '{print $8}' training_stats.csv | sort | uniq -c

# Sprawdź średnią długość
awk -F',' 'NR>1 {sum+=$7; count++} END {print sum/count}' training_stats.csv
```

Powodzenia! 🐍
