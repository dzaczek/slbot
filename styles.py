
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
                "gamma": 0.8670558109508336,
                "food_reward": 4.694645749412523,
                "food_shaping": 0.7445652175384821,
                "survival": 0.06791100522162087,
                "death_wall": -14.281573613004564,
                "death_snake": -15.790517187647083,
                "wall_proximity_penalty": 0.055092764826992865,
                "max_steps": 593,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.91275855060423,
                "food_reward": 6.580895930371094,
                "food_shaping": 0.0802427598183686,
                "survival": 0.23943042394886488,
                "survival_escalation": 0.002823234020298721,
                "death_wall": -33.95449838255479,
                "death_snake": -27.54636510634126,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 2.040366145191224,
                "starvation_penalty": 0.0018310966325919666,
                "starvation_grace_steps": 40,
                "max_steps": 618,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.9039365673580718,
                "food_reward": 3.006196967829972,
                "food_shaping": 0.06428597239109865,
                "survival": 0.2310100682349972,
                "death_wall": -25.30177262062398,
                "death_snake": -49.632870430701956,
                "enemy_alert_dist": 1777,
                "enemy_proximity_penalty": 0.9430327673571719,
                "enemy_approach_penalty": 0.8701223408214269,
                "boost_penalty": 0.14749219484757445,
                "starvation_penalty": 0.005563157911856432,
                "starvation_grace_steps": 61,
                "max_steps": 1642,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9580632336902466,
                "food_reward": 7.500615593605694,
                "food_shaping": 0.14546967157234186,
                "survival": 0.11844046246097659,
                "survival_escalation": 0.004239936773214511,
                "death_wall": -28.87991442555279,
                "death_snake": -46.17879063460386,
                "length_bonus": 0.04637333776719822,
                "wall_proximity_penalty": 0.2953273095727919,
                "enemy_alert_dist": 2679,
                "enemy_proximity_penalty": 1.3429003886009345,
                "enemy_approach_penalty": 0.1335883940213632,
                "boost_penalty": 4.821378747025808,
                "mass_loss_penalty": 4.377025344741382,
                "starvation_penalty": 0.08948240015697684,
                "starvation_grace_steps": 21,
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
                "food_reward": 6.461680558143879,
                "food_shaping": 0.07353716494912636,
                "survival": 0.21593242258062362,
                "survival_escalation": 0.0009618378800855905,
                "death_wall": -38.313800614279046,
                "death_snake": -34.696774782237576,
                "length_bonus": 0.014988514306005975,
                "wall_proximity_penalty": 0.288976020257623,
                "enemy_alert_dist": 2017,
                "enemy_proximity_penalty": 1.2463959370106568,
                "enemy_approach_penalty": 0.24505702834255474,
                "boost_penalty": 4.0,
                "mass_loss_penalty": 2.522239111324971,
                "starvation_penalty": 0.015864734511593752,
                "starvation_grace_steps": 98,
                "contest_food_reward": 1.4541077626865913,
                "enemy_zone_control_reward": 0.06358149847709144,
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
