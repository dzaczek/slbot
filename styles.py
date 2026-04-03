
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
                "food_reward": 1.2747836417165774,
                "food_shaping": 0.45959485007350204,
                "survival": 0.0965369660749581,
                "death_wall": -12.655313449148945,
                "death_snake": -11.205889886774356,
                "wall_proximity_penalty": 0.02333848419540233,
                "max_steps": 494,
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
                "food_shaping": 0.07685110168673592,
                "survival": 0.33593474126736345,
                "survival_escalation": 0.006341205168159339,
                "death_wall": -48.71041902753709,
                "death_snake": -26.918950105702727,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 0.7751874757208957,
                "starvation_penalty": 0.0047277766988262065,
                "starvation_grace_steps": 33,
                "max_steps": 409,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.9038914892875372,
                "food_reward": 4.537006169330944,
                "food_shaping": 0.0899568440140703,
                "survival": 0.30601584481372435,
                "death_wall": -12.031636783433125,
                "death_snake": -61.86474242352617,
                "enemy_alert_dist": 2367,
                "enemy_proximity_penalty": 1.283873478349702,
                "enemy_approach_penalty": 0.6948387383732386,
                "boost_penalty": 0.11439340742426059,
                "starvation_penalty": 0.0012208741801798446,
                "starvation_grace_steps": 39,
                "max_steps": 950,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9580632336902466,
                "food_reward": 9.442317584327691,
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
                "gamma": 0.9917980637588021,
                "food_reward": 4.600110781769387,
                "food_shaping": 0.09269246289894084,
                "survival": 0.07936050326806246,
                "survival_escalation": 0.001906910005444593,
                "death_wall": -38.66775079390848,
                "death_snake": -23.369761985892794,
                "length_bonus": 0.014988514306005975,
                "wall_proximity_penalty": 0.5171845942225712,
                "enemy_alert_dist": 1864,
                "enemy_proximity_penalty": 1.6428445361702049,
                "enemy_approach_penalty": 0.12578192127266835,
                "boost_penalty": 5.885375789499927,
                "mass_loss_penalty": 1.8904595251282406,
                "starvation_penalty": 0.015864734511593752,
                "starvation_grace_steps": 138,
                "contest_food_reward": 2.2233061779486896,
                "enemy_zone_control_reward": 0.05499565440155462,
                "kill_opportunity_reward": 7.831226933823622,
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
