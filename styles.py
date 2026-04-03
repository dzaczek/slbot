
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
                "gamma": 0.8533937385879289,
                "food_reward": 1.2924219102248489,
                "food_shaping": 0.492083557602094,
                "survival": 0.09506689948133075,
                "death_wall": -14.521452741959445,
                "death_snake": -10.783515128084249,
                "wall_proximity_penalty": 0.03433588034632636,
                "max_steps": 652,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.8087524144554349,
                "food_reward": 7.277976571910434,
                "food_shaping": 0.06182106927248518,
                "survival": 0.2641553072549348,
                "survival_escalation": 0.01,
                "death_wall": -34.422390381545156,
                "death_snake": -19.810610826600456,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 2.2754597604514313,
                "starvation_penalty": 0.0047277766988262065,
                "starvation_grace_steps": 33,
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
                "survival": 0.2997688982799574,
                "death_wall": -12.031636783433125,
                "death_snake": -63.03412956163926,
                "enemy_alert_dist": 2367,
                "enemy_proximity_penalty": 0.9430327673571719,
                "enemy_approach_penalty": 0.5168210048707321,
                "boost_penalty": 0.10402916544480031,
                "starvation_penalty": 0.0005782204068036805,
                "starvation_grace_steps": 34,
                "max_steps": 1242,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9580632336902466,
                "food_reward": 7.590035121450558,
                "food_shaping": 0.09416480866663768,
                "survival": 0.08341992041932669,
                "survival_escalation": 0.004239936773214511,
                "death_wall": -28.956838102785166,
                "death_snake": -40.52883369869288,
                "length_bonus": 0.06464836307992215,
                "wall_proximity_penalty": 0.2953273095727919,
                "enemy_alert_dist": 2679,
                "enemy_proximity_penalty": 1.3429003886009345,
                "enemy_approach_penalty": 0.1335883940213632,
                "boost_penalty": 2.9362928230263377,
                "mass_loss_penalty": 1.84921905444158,
                "starvation_penalty": 0.07820260580939312,
                "starvation_grace_steps": 21,
                "max_steps": 2240,
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
                "food_reward": 4.600110781769387,
                "food_shaping": 0.09269246289894084,
                "survival": 0.13662993066192516,
                "survival_escalation": 0.0012221637332453775,
                "death_wall": -39.08900371864204,
                "death_snake": -27.902312782724408,
                "length_bonus": 0.014988514306005975,
                "wall_proximity_penalty": 0.3399122430344894,
                "enemy_alert_dist": 1678,
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
