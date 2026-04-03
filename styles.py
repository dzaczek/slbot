
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
                "gamma": 0.8835107739480733,
                "food_reward": 2.9934567884274266,
                "food_shaping": 1.0,
                "survival": 0.08996534715094971,
                "death_wall": -17.116233983284246,
                "death_snake": -18.213535398809373,
                "wall_proximity_penalty": 0.3724944703765466,
                "max_steps": 814,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.9010414262889107,
                "food_reward": 6.798689328431706,
                "food_shaping": 0.12778240916814104,
                "survival": 0.2948802908591791,
                "survival_escalation": 0.002823234020298721,
                "death_wall": -31.532415773808637,
                "death_snake": -28.941751133194487,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 2.1109383340987415,
                "starvation_penalty": 0.003536489483081666,
                "starvation_grace_steps": 70,
                "max_steps": 539,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.9368366113547488,
                "food_reward": 5.381336268785208,
                "food_shaping": 0.09691015314752094,
                "survival": 0.3,
                "death_wall": -40,
                "death_snake": -39.59706296492766,
                "enemy_alert_dist": 1777,
                "enemy_proximity_penalty": 1.0073477545610467,
                "enemy_approach_penalty": 0.6130696281209953,
                "boost_penalty": 0.11505568937057176,
                "starvation_penalty": 0.008670972600010488,
                "starvation_grace_steps": 60,
                "max_steps": 2000,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9580632336902466,
                "food_reward": 7.227190146114899,
                "food_shaping": 0.12536875497982988,
                "survival": 0.11859010503691997,
                "survival_escalation": 0.002582270391375317,
                "death_wall": -39.41001688408666,
                "death_snake": -40.742732177312234,
                "length_bonus": 0.02737293013487195,
                "wall_proximity_penalty": 0.445380341197032,
                "enemy_alert_dist": 2679,
                "enemy_proximity_penalty": 1.3429003886009345,
                "enemy_approach_penalty": 0.2948955039980742,
                "boost_penalty": 4.083701277013601,
                "mass_loss_penalty": 4.377025344741382,
                "starvation_penalty": 0.09786744526136462,
                "starvation_grace_steps": 28,
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
                "gamma": 0.977149389635851,
                "food_reward": 5.179832807909898,
                "food_shaping": 0.16241472434957416,
                "survival": 0.21593242258062362,
                "survival_escalation": 0.0009618378800855905,
                "death_wall": -38.313800614279046,
                "death_snake": -45.24874538535382,
                "length_bonus": 0.025117472685273715,
                "wall_proximity_penalty": 0.288976020257623,
                "enemy_alert_dist": 1580,
                "enemy_proximity_penalty": 1.2463959370106568,
                "enemy_approach_penalty": 0.24505702834255474,
                "boost_penalty": 4.0,
                "mass_loss_penalty": 2.808368369842686,
                "starvation_penalty": 0.018429804708753286,
                "starvation_grace_steps": 112,
                "contest_food_reward": 1.4541077626865913,
                "enemy_zone_control_reward": 0.08397733750405592,
                "kill_opportunity_reward": 14.033302860502154,
                "max_steps": 3148,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 1200,
                    "avg_peak_length": 100
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
