
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
                "gamma": 0.8490574889899353,
                "food_reward": 2.134116092099309,
                "food_shaping": 1.0,
                "survival": 0.09268929949607274,
                "death_wall": -16.57154124490194,
                "death_snake": -16.667142860418856,
                "wall_proximity_penalty": 0.02872535240206913,
                "max_steps": 782,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.8401793947682854,
                "food_reward": 7.277976571910434,
                "food_shaping": 0.07177149499650093,
                "survival": 0.1753408874368828,
                "survival_escalation": 0.0073896103873032455,
                "death_wall": -34.422390381545156,
                "death_snake": -26.313713449365338,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 2.4931548294569668,
                "starvation_penalty": 0.0022196008750340673,
                "starvation_grace_steps": 40,
                "max_steps": 461,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.8646479633059637,
                "food_reward": 3.238740609541266,
                "food_shaping": 0.0985389088145585,
                "survival": 0.2310100682349972,
                "death_wall": -14.012858459642402,
                "death_snake": -71.0700171079272,
                "enemy_alert_dist": 2367,
                "enemy_proximity_penalty": 0.9430327673571719,
                "enemy_approach_penalty": 0.9606383308983857,
                "boost_penalty": 0.10402916544480031,
                "starvation_penalty": 0.0015103829640965049,
                "starvation_grace_steps": 34,
                "max_steps": 1480,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9580632336902466,
                "food_reward": 7.500615593605694,
                "food_shaping": 0.09416480866663768,
                "survival": 0.11844046246097659,
                "survival_escalation": 0.004239936773214511,
                "death_wall": -28.956838102785166,
                "death_snake": -40.52883369869288,
                "length_bonus": 0.06464836307992215,
                "wall_proximity_penalty": 0.2953273095727919,
                "enemy_alert_dist": 2679,
                "enemy_proximity_penalty": 1.3429003886009345,
                "enemy_approach_penalty": 0.1335883940213632,
                "boost_penalty": 4.621169709321238,
                "mass_loss_penalty": 1.84921905444158,
                "starvation_penalty": 0.07820260580939312,
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
                "gamma": 0.9641177884845898,
                "food_reward": 5.400390132588768,
                "food_shaping": 0.09269246289894084,
                "survival": 0.13662993066192516,
                "survival_escalation": 0.0012221637332453775,
                "death_wall": -39.08900371864204,
                "death_snake": -27.902312782724408,
                "length_bonus": 0.014988514306005975,
                "wall_proximity_penalty": 0.3399122430344894,
                "enemy_alert_dist": 1804,
                "enemy_proximity_penalty": 1.0976617909670998,
                "enemy_approach_penalty": 0.2621917853155132,
                "boost_penalty": 5.20709691062441,
                "mass_loss_penalty": 2.522239111324971,
                "starvation_penalty": 0.015864734511593752,
                "starvation_grace_steps": 138,
                "contest_food_reward": 2.2233061779486896,
                "enemy_zone_control_reward": 0.06358149847709144,
                "kill_opportunity_reward": 6.031317975974872,
                "max_steps": 3245,
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
