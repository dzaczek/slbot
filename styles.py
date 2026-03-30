
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
                "death_wall": -15,
                "death_snake": -15,
                "wall_proximity_penalty": 0.3,
                "max_steps": 600,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
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
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 1.5,
                "starvation_penalty": 0.005,
                "starvation_grace_steps": 80,
                "max_steps": 500,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.95,
                "food_reward": 5.0,
                "food_shaping": 0.1,
                "survival": 0.3,
                "death_wall": -40,
                "death_snake": -40,
                "enemy_alert_dist": 2000,
                "enemy_proximity_penalty": 1.5,
                "enemy_approach_penalty": 0.5,
                "boost_penalty": 0.1,
                "starvation_penalty": 0.008,
                "starvation_grace_steps": 60,
                "max_steps": 2000,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.97,
                "food_reward": 8.0,
                "food_shaping": 0.3,
                "survival": 0.2,
                "survival_escalation": 0.001,
                "death_wall": -35,
                "death_snake": -50,
                "length_bonus": 0.02,
                "wall_proximity_penalty": 0.5,
                "enemy_alert_dist": 2000,
                "enemy_proximity_penalty": 1.2,
                "enemy_approach_penalty": 0.8,
                "boost_penalty": 4.0,
                "mass_loss_penalty": 6.0,
                "starvation_penalty": 0.1,
                "starvation_grace_steps": 20,
                "max_steps": 2000,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 600,
                    "avg_peak_length": 40
                },
                "promote_window": 200
            },
            5: {
                "name": "MASTERY_SURVIVAL",
                "gamma": 0.97,
                "food_reward": 8.0,
                "food_shaping": 0.25,
                "survival": 0.3323001565477235,
                "survival_escalation": 0.0005,
                "death_wall": -45,
                "death_snake": -50,
                "length_bonus": 0.05,
                "wall_proximity_penalty": 0.35,
                "enemy_alert_dist": 2051,
                "enemy_proximity_penalty": 1.0,
                "enemy_approach_penalty": 0.4,
                "boost_penalty": 4.0,
                "mass_loss_penalty": 4.932413965597075,
                "starvation_penalty": 0.014849148109343756,
                "starvation_grace_steps": 100,
                "contest_food_reward": 1.0,
                "enemy_zone_control_reward": 0.05,
                "kill_opportunity_reward": 15.0,
                "max_steps": 3000,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 1500,
                    "avg_peak_length": 80
                },
                "promote_window": 500
            },
            6: {
                "name": "APEX_PREDATOR",
                "gamma": 0.99,
                "food_reward": 8.0,
                "survival": 0.1,
                "death_wall": -40,
                "death_snake": -30,
                "enemy_alert_dist": 2000,
                "enemy_proximity_penalty": 0.15,
                "boost_penalty": 0.0,
                "contest_food_reward": 1.0,
                "enemy_zone_control_reward": 0.06,
                "kill_opportunity_reward": 18.0,
                "max_steps": 99999,
                "promote_metric": None
            }
        }
    },
    "Aggressive (Hunter)": {
        "type": "static",
        "description": "High reward for eating. Low survival bonus.",
        "config": {
            "name": "HUNTER",
            "food_reward": 20.0,
            "food_shaping": 0.05,
            "survival": 0.0,
            "death_wall": -50,
            "death_snake": -10,
            "wall_proximity_penalty": 0.05,
            "enemy_proximity_penalty": 0.05,
            "max_steps": 99999
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
            "wall_proximity_penalty": 0.2,
            "enemy_proximity_penalty": 0.15,
            "max_steps": 99999
        }
    },
    "Explorer (Anti-Float)": {
        "type": "static",
        "description": "Penalizes staying still. Forces movement.",
        "config": {
            "name": "EXPLORER",
            "food_reward": 5.0,
            "food_shaping": 0.05,
            "survival": 0.05,
            "death_wall": -50,
            "death_snake": -40,
            "straight_penalty": 0.1,
            "wall_proximity_penalty": 0.5,
            "enemy_proximity_penalty": 0.1,
            "max_steps": 99999
        }
    }
}
