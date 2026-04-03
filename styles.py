
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
                "gamma": 0.9059096275906025,
                "food_reward": 1.0149038742700327,
                "food_shaping": 0.492083557602094,
                "survival": 0.10482439133051549,
                "death_wall": -10.75918497960625,
                "death_snake": -13.216302531681142,
                "wall_proximity_penalty": 0.02239069838875144,
                "max_steps": 474,
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
                "food_reward": 7.578467424173228,
                "food_shaping": 0.0716377114178595,
                "survival": 0.33593474126736345,
                "survival_escalation": 0.01,
                "death_wall": -43.62469062585754,
                "death_snake": -26.918950105702727,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 0.7751874757208957,
                "starvation_penalty": 0.0047277766988262065,
                "starvation_grace_steps": 33,
                "max_steps": 411,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.9038914892875372,
                "food_reward": 4.537006169330944,
                "food_shaping": 0.0985389088145585,
                "survival": 0.29704825587844663,
                "death_wall": -12.031636783433125,
                "death_snake": -76.99046682777795,
                "enemy_alert_dist": 2367,
                "enemy_proximity_penalty": 0.9430327673571719,
                "enemy_approach_penalty": 0.6948387383732386,
                "boost_penalty": 0.11439340742426059,
                "starvation_penalty": 0.0009765722051477506,
                "starvation_grace_steps": 34,
                "max_steps": 950,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9580632336902466,
                "food_reward": 8.788800980046663,
                "food_shaping": 0.09416480866663768,
                "survival": 0.08341992041932669,
                "survival_escalation": 0.004239936773214511,
                "death_wall": -28.956838102785166,
                "death_snake": -49.51663138235109,
                "length_bonus": 0.06464836307992215,
                "wall_proximity_penalty": 0.2953273095727919,
                "enemy_alert_dist": 2679,
                "enemy_proximity_penalty": 1.3429003886009345,
                "enemy_approach_penalty": 0.14349671937965644,
                "boost_penalty": 3.304300286573684,
                "mass_loss_penalty": 1.84921905444158,
                "starvation_penalty": 0.1093577504335424,
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
