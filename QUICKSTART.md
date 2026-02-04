# Quick Start Guide

## 🚀 Getting Started

### 1. Installation (One-time)
```bash
pip install -r requirements.txt
```

### 2. System Test (Optional but Recommended)
```bash
python quick_test.py
```
Runs the bot for 30 seconds to verify everything works.

### 3. Training

**Option A: Interactive restart (recommended)**
```bash
./restart_training.sh
```
- Choose to continue or start fresh
- Automatically creates backups

**Option B: Direct start**
```bash
python training_manager.py
```

### 4. Watch the Best Bot
```bash
python play_best.py
```

### 5. Analyze Progress
```bash
python analyze_training.py
```

---

## 📋 All Available Commands

### Training
| Command | Description |
|---------|-------------|
| `./restart_training.sh` | Interactive restart with backup option |
| `python training_manager.py` | Start/resume training |
| `python quick_test.py` | 30s test (system verification) |

### Playing
| Command | Description |
|---------|-------------|
| `python play_best.py` | Play with the best genome |
| `python play_best.py neat-checkpoint-50` | Play genome from checkpoint |

### Analysis
| Command | Description |
|---------|-------------|
| `python analyze_training.py` | Show training statistics |
| `python analyze_training.py --live` | Live mode with auto-refresh |
| `tail -f training_log.txt` | Follow logs in real-time |
| `tail -50 training_stats.csv` | Last 50 results |

---

## 🎯 What to Expect

### Generations 1-10
- Bot will be dumb, die quickly
- Some will start eating 1-3 food
- Fitness: ~160 → ~400

### Generations 10-30
- Bot starts eating regularly
- 5-15 food per life
- Fitness: ~400 → ~1000

### Generations 30-50
- Bot avoids walls and enemies
- 15-30 food
- Fitness: ~1000 → ~2000+

### Generations 50+
- Bot is smart!
- Long survival, lots of food
- Fitness: 2000+

---

## 🔧 Troubleshooting

### Bot doesn't eat food after 50 generations?
```bash
# Increase food reward in training_manager.py, line ~214:
fitness_score += (diff * 200.0)  # Was 150.0
```

### Bot keeps hitting walls?
```bash
# In spatial_awareness.py, line ~209, increase boost:
wall_danger = min(wall_danger * 2.0, 1.0)  # Was 1.5
```

### Training is too slow?
```bash
# In training_manager.py, line ~442, change:
NUM_WORKERS = 3  # Reduce if computer is slow
```

### Chrome doesn't open?
```bash
# Check if Chrome is installed
# Install webdriver-manager:
pip install webdriver-manager
```

---

## 📊 Parameters for Tuning

Open `training_manager.py` and find these values:

```python
# Line ~214 - Food reward
fitness_score += (diff * 150.0)  # Increase = more aggressive eating

# Line ~218 - Starvation timeout
if time.time() - last_eat_time > 60:  # Increase = more time to find food

# Line ~266 - Length reward
fitness_score += (max_len * 20)  # Increase = more motivation to grow

# Line ~263 - Survival time weight
fitness_score += (survival_time * 2.0)  # Decrease = less passive survival
```

---

## 🎓 Pro Tips

1. **Start Fresh After Changes**: Old genomes are "stuck" in bad habits
2. **Monitor Logs**: `tail -f training_log.txt` shows what's happening
3. **Backup Often**: Best genomes can be overwritten
4. **Patience**: A good bot needs 50-100+ generations
5. **Experiment**: Change parameters and see what works!

---

## 📁 Important Files

| File | Description |
|------|-------------|
| `training_manager.py` | Main training loop + fitness |
| `config_neat.txt` | NEAT parameters (mutations, population) |
| `spatial_awareness.py` | Game data processing |
| `ai_brain.py` | Neural network wrapper |
| `browser_engine.py` | Browser control |
| `best_genome.pkl` | Best trained genome |
| `neat-checkpoint-X` | Checkpoints (auto-save) |
| `training_stats.csv` | History of all evaluations |

---

## ❓ FAQ

**Q: How long does training take?**  
A: 50 generations × 50 genomes = 2500 evaluations. With 5 workers ~1-2 hours.

**Q: Can I interrupt and resume?**  
A: Yes! Ctrl+C and then `python training_manager.py` will resume from last checkpoint.

**Q: How to save the best bot?**  
A: Automatically saved as `best_genome.pkl` at the end of training.

**Q: Can I train without browser window?**  
A: Yes, in `training_manager.py` set `HEADLESS = True` (line ~443).

**Q: Bot is too defensive/aggressive?**  
A: Change balance between survival (line 263) and food rewards (line 214).

---

Good luck! 🐍🎮
