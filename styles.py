
"""
Reward definitions for different learning styles.
"""

STYLES = {
    "Standard (Curriculum)": {
        "type": "curriculum",
        "description": "Progressive learning: Food -> Walls -> Enemies -> Strategy",
        "stages": {
            1: {
                "name": "FOOD_VECTOR",
                "gamma": 0.85,
                "food_reward": 3.0,
                "food_shaping": 0.5,
                "survival": 0.1,
                "survival_escalation": 0.0,
                "death_wall": -15,
                "death_snake": -15,
                "straight_penalty": 0.0,
                "length_bonus": 0.0,
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 800,
                "wall_proximity_penalty": 0.3,
                "enemy_proximity_penalty": 0.0,
                "enemy_approach_penalty": 0.0,
                "boost_penalty": 0.0,
                "max_steps": 600,
                # COMPOUND: must eat AND survive — require solid food-seeking skill
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80,
                },
                "promote_window": 400,
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.93,
                "food_reward": 5.0,
                "food_shaping": 0.15,
                "survival": 0.3,
                "survival_escalation": 0.002,
                "death_wall": -40,
                "death_snake": -20,
                "straight_penalty": 0.0,
                "length_bonus": 0.0,
                "wall_alert_dist": 2500,
                "enemy_alert_dist": 800,
                "wall_proximity_penalty": 1.5,
                "enemy_proximity_penalty": 0.0,
                "enemy_approach_penalty": 0.0,
                "boost_penalty": 0.0,
                "starvation_penalty": 0.005,
                "starvation_grace_steps": 80,
                "starvation_max_penalty": 1.0,
                "max_steps": 500,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.10,
                "promote_window": 400,
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.95,
                "food_reward": 5.0,
                "food_shaping": 0.1,
                "survival": 0.3,
                "survival_escalation": 0.001,
                "death_wall": -40,
                "death_snake": -40,
                "straight_penalty": 0.0,
                "length_bonus": 0.0,
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 2000,
                "wall_proximity_penalty": 0.5,
                "enemy_proximity_penalty": 1.5,
                "enemy_approach_penalty": 0.5,
                "boost_penalty": 0.1,
                "starvation_penalty": 0.008,
                "starvation_grace_steps": 60,
                "starvation_max_penalty": 1.5,
                "max_steps": 2000,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500,
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.97,
                "food_reward": 8.0,               # was 5.0 — stronger food drive to grow mass
                "food_shaping": 0.15,             # was 0.1 — more reward for approaching food
                "survival": 0.2,
                "survival_escalation": 0.001,
                "death_wall": -35,
                "death_snake": -25,
                "straight_penalty": 0.0,
                "length_bonus": 0.2,              # strong reward for being big (size 100 = +20/step)
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 1000,
                "wall_proximity_penalty": 0.5,
                "enemy_proximity_penalty": 0.8,
                "enemy_approach_penalty": 0.3,
                "boost_penalty": 1.0,              # discourage burning mass via boost
                "starvation_penalty": 0.03,       # was 0.01 — stronger hunger pressure
                "starvation_grace_steps": 40,     # was 50 — less grace before penalty kicks in
                "starvation_max_penalty": 3.0,    # was 2.0 — harder cap on starvation
                "max_steps": 2000,
                # Promote to S5 when avg_steps >= 1000 over 500 eps
                "promote_metric": "avg_steps",
                "promote_threshold": 1000,
                "promote_window": 500,
            },
            5: {
                "name": "MASTERY_SURVIVAL",
                "gamma": 0.99,                    # far-sighted: long-term survival matters
                "food_reward": 8.0,               # strong food drive — grow big
                "food_shaping": 0.05,             # mild approach shaping (don't chase blindly)
                "survival": 0.4,                  # high per-step survival reward
                "survival_escalation": 0.0005,    # gentle escalation — reward living longer
                "death_wall": -45,                # wall death severely punished
                "death_snake": -50,               # snake collision = worst outcome
                "straight_penalty": 0.0,
                "length_bonus": 0.05,             # strong reward for being big
                "wall_alert_dist": 2500,
                "enemy_alert_dist": 2500,         # wide enemy radar — avoid early
                "wall_proximity_penalty": 0.8,
                "enemy_proximity_penalty": 2.0,   # very high — stay far from enemies
                "enemy_approach_penalty": 1.0,    # heavily penalize closing distance
                "boost_penalty": 0.2,             # discourage risky boost (save mass)
                "starvation_penalty": 0.008,
                "starvation_grace_steps": 100,    # more grace — longer episodes
                "starvation_max_penalty": 1.5,
                "max_steps": 99999,               # no artificial limit
                # Promote to S6 when surviving avg 3500 steps over 500 eps
                "promote_metric": "avg_steps",
                "promote_threshold": 3500,
                "promote_window": 500,
            },
            6: {
                "name": "APEX_PREDATOR",
                "gamma": 0.99,
                "food_reward": 10.0,              # max food reward — consume kills
                "food_shaping": 0.03,             # mild shaping
                "survival": 0.3,                  # moderate survival (aggression > safety)
                "survival_escalation": 0.0003,
                "death_wall": -40,
                "death_snake": -30,               # reduced snake death penalty — accept combat risk
                "straight_penalty": 0.0,
                "length_bonus": 0.03,             # still rewarded for mass
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 1500,         # narrower radar — approach enemies
                "wall_proximity_penalty": 0.5,
                "enemy_proximity_penalty": 0.3,   # low — don't fear enemies
                "enemy_approach_penalty": 0.0,    # zero — closing in is OK
                "boost_penalty": 0.0,             # boost is free — use it to cut off enemies
                "starvation_penalty": 0.012,      # push to hunt, not idle
                "starvation_grace_steps": 80,
                "starvation_max_penalty": 2.5,
                "max_steps": 99999,               # no limit — final stage
                "promote_metric": None,           # terminal stage
                "promote_threshold": None,
                "promote_window": 100,
            },
        }
    },
    "Aggressive (Hunter)": {
        "type": "static",
        "description": "High reward for eating and moving towards food. Low survival bonus.",
        "config": {
            "name": "HUNTER",
            "food_reward": 20.0,
            "food_shaping": 0.05,
            "survival": 0.0,
            "death_wall": -50,
            "death_snake": -10, # Risk taking allowed
            "straight_penalty": 0.0,
            "length_bonus": 0.0,
            "wall_alert_dist": 1500,
            "enemy_alert_dist": 700,
            "wall_proximity_penalty": 0.05,
            "enemy_proximity_penalty": 0.05,
            "max_steps": 99999,
        }
    },
    "Defensive (Safe)": {
        "type": "static",
        "description": "High survival bonus and heavy death penalties.",
        "config": {
            "name": "SAFE",
            "food_reward": 5.0,
            "food_shaping": 0.005,
            "survival": 0.5,
            "death_wall": -50,
            "death_snake": -40,
            "straight_penalty": 0.05,
            "length_bonus": 0.0,
            "wall_alert_dist": 2200,
            "enemy_alert_dist": 900,
            "wall_proximity_penalty": 0.2,
            "enemy_proximity_penalty": 0.15,
            "max_steps": 99999,
        }
    },
    "Explorer (Anti-Float)": {
        "type": "static",
        "description": "Penalizes staying still/floating. Forces movement.",
        "config": {
            "name": "EXPLORER",
            "food_reward": 5.0,
            "food_shaping": 0.05,   # Strongly encourage seeking food
            "survival": 0.05,
            "death_wall": -50,
            "death_snake": -40,
            "straight_penalty": 0.1, # Force turning/activity
            "length_bonus": 0.0,
            "wall_alert_dist": 1800,
            "enemy_alert_dist": 800,
            "wall_proximity_penalty": 0.5,
            "enemy_proximity_penalty": 0.1,
            "max_steps": 99999,
        }
    }
}
