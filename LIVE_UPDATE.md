
## Live Update (Hot-Reload)

System SlitherBot obsługuje dynamiczną zmianę parametrów nagród bez restartowania procesu treningu:

1. **Ręczny Update (styles.py):** Możesz edytować plik `styles.py` w trakcie działania bota. Po każdym ukończonym epizodzie trener sprawdza datę modyfikacji pliku i jeśli został zmieniony, automatycznie przeładowuje wagi nagród (`food_reward`, `boost_penalty`, `proximity_penalty` itd.).
2. **Automatyczny Update (config_ai.json):** Jeśli używasz flagi `--ai-supervisor`, system będzie generował i odczytywał plik `config_ai.json` z rekomendacjami od AI, nadpisując parametry w locie.

Dzięki temu możesz na bieżąco korygować zachowanie agenta (np. gdy zauważysz, że zbyt często używa boosta), obserwując efekty w następnych epizodach.
