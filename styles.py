
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
                "gamma": 0.8749862442079119,
                "food_reward": 1.823701758796462,
                "food_shaping": 0.8568033308602445,
                "survival": 0.06791100522162087,
                "death_wall": -13.974925630149672,
                "death_snake": -13.180286899548523,
                "wall_proximity_penalty": 0.062063623779473484,
                "max_steps": 634,
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
                "food_shaping": 0.11513801486982633,
                "survival": 0.13521839798720178,
                "survival_escalation": 0.003106537671083698,
                "death_wall": -33.95449838255479,
                "death_snake": -25.764551706028108,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 2.040366145191224,
                "starvation_penalty": 0.0018310966325919666,
                "starvation_grace_steps": 40,
                "max_steps": 659,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.8646479633059637,
                "food_reward": 3.006196967829972,
                "food_shaping": 0.0985389088145585,
                "survival": 0.2310100682349972,
                "death_wall": -16.12936577330337,
                "death_snake": -71.0700171079272,
                "enemy_alert_dist": 2367,
                "enemy_proximity_penalty": 0.9430327673571719,
                "enemy_approach_penalty": 1.0021844484201212,
                "boost_penalty": 0.11944453377651425,
                "starvation_penalty": 0.0008125027326997339,
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
                "death_wall": -28.87991442555279,
                "death_snake": -37.56901463506555,
                "length_bonus": 0.06464836307992215,
                "wall_proximity_penalty": 0.2953273095727919,
                "enemy_alert_dist": 2679,
                "enemy_proximity_penalty": 1.3429003886009345,
                "enemy_approach_penalty": 0.1335883940213632,
                "boost_penalty": 4.621169709321238,
                "mass_loss_penalty": 3.014171514851726,
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
