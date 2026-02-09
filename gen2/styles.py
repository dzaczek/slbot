
"""
Reward definitions for different learning styles.
"""

STYLES = {
    "Standard (Curriculum)": {
        "type": "curriculum",
        "description": "Progressive learning: Eat -> Survive -> Grow",
        "stages": {
            1: {
                "name": "EAT",
                "food_reward": 8.0,
                "food_shaping": 0.015,  # Increased to fix floating
                "survival": 0.05,
                "death_wall": -200,     # Increased to prevent suicide eating
                "death_snake": -100,
                "straight_penalty": 0.0,
                "length_bonus": 0.0,
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 800,
                "wall_proximity_penalty": 0.0,
                "enemy_proximity_penalty": 0.0,
                "max_steps": 300,
                "promote_metric": "food_per_step",
                "promote_threshold": 0.06,
                "promote_window": 50,
            },
            2: {
                "name": "SURVIVE",
                "food_reward": 8.0,
                "food_shaping": 0.01,
                "survival": 0.1,
                "death_wall": -100,
                "death_snake": -30,
                "straight_penalty": 0.02,
                "length_bonus": 0.0,
                "wall_alert_dist": 1800,
                "enemy_alert_dist": 700,
                "wall_proximity_penalty": 0.15,
                "enemy_proximity_penalty": 0.1,
                "max_steps": 800,
                "promote_metric": "avg_steps",
                "promote_threshold": 250,
                "promote_window": 50,
            },
            3: {
                "name": "GROW",
                "food_reward": 10.0,
                "food_shaping": 0.01,
                "survival": 0.1,
                "death_wall": -100,
                "death_snake": -30,
                "straight_penalty": 0.02,
                "length_bonus": 0.02,
                "wall_alert_dist": 1800,
                "enemy_alert_dist": 700,
                "wall_proximity_penalty": 0.1,
                "enemy_proximity_penalty": 0.08,
                "max_steps": 99999,
                "promote_metric": None,
                "promote_threshold": None,
                "promote_window": 50,
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
            "death_wall": -200,
            "death_snake": -100,
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
            "food_shaping": 0.02,
            "survival": 0.05,
            "death_wall": -200,
            "death_snake": -100,
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
