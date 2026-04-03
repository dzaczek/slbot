
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
                "gamma": 0.8457625654118586,
                "food_reward": 1.0140964625560098,
                "food_shaping": 0.5719178655206834,
                "survival": 0.03487830264685447,
                "death_wall": -13.950323078987699,
                "death_snake": -12.536147281830095,
                "wall_proximity_penalty": 0.023925459890302817,
                "max_steps": 560,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.8,
                "food_reward": 2.366231438797426,
                "food_shaping": 0.08328493683804591,
                "survival": 0.26310692913884864,
                "survival_escalation": 0.007433119730524345,
                "death_wall": -38.96679271845376,
                "death_snake": -18.867020680429135,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 0.24682696407601323,
                "starvation_penalty": 0.005743137069825212,
                "starvation_grace_steps": 31,
                "max_steps": 323,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.9038914892875372,
                "food_reward": 3.7534694785699796,
                "food_shaping": 0.0899568440140703,
                "survival": 0.30545305396530975,
                "death_wall": -5,
                "death_snake": -44.215265401691326,
                "enemy_alert_dist": 2367,
                "enemy_proximity_penalty": 1.263942747222119,
                "enemy_approach_penalty": 0.6948387383732386,
                "boost_penalty": 0.07557440472126242,
                "starvation_penalty": 0.0011414678909526773,
                "starvation_grace_steps": 42,
                "max_steps": 842,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9202738140825181,
                "food_reward": 9.442317584327691,
                "food_shaping": 0.0569611935505282,
                "survival": 0.08341992041932669,
                "survival_escalation": 0.005522935510440249,
                "death_wall": -33.44811494557678,
                "death_snake": -63.34463454640907,
                "length_bonus": 0.06464836307992215,
                "wall_proximity_penalty": 0.22969152856724978,
                "enemy_alert_dist": 3225,
                "enemy_proximity_penalty": 1.3429003886009345,
                "enemy_approach_penalty": 0.14349671937965644,
                "boost_penalty": 3.304300286573684,
                "mass_loss_penalty": 1.7527644650249679,
                "starvation_penalty": 0.1093577504335424,
                "starvation_grace_steps": 23,
                "max_steps": 2526,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 600,
                    "avg_peak_length": 40
                },
                "promote_window": 200
            },
            5: {
                "name": "MASTERY_SURVIVAL",
                "gamma": 0.995,
                "food_reward": 3.878435206593664,
                "food_shaping": 0.09269246289894084,
                "survival": 0.07936050326806246,
                "survival_escalation": 0.0019461765396040036,
                "death_wall": -27.690783206824932,
                "death_snake": -31.687700422432613,
                "length_bonus": 0.0060892529642129,
                "wall_proximity_penalty": 0.38132229474966967,
                "enemy_alert_dist": 1471,
                "enemy_proximity_penalty": 1.4242872203165329,
                "enemy_approach_penalty": 0.10430185150108015,
                "boost_penalty": 6.047500084734265,
                "mass_loss_penalty": 1.8904595251282406,
                "starvation_penalty": 0.011613285062345293,
                "starvation_grace_steps": 138,
                "contest_food_reward": 2.819182323729337,
                "enemy_zone_control_reward": 0.06585261178978555,
                "kill_opportunity_reward": 7.831226933823622,
                "max_steps": 4035,
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
