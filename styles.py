
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
                "survival_escalation": 0.0,
                "death_wall": -15,
                "death_snake": -15,
                "straight_penalty": 0.0,
                "length_bonus": 0.0,
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 800,
                "wall_proximity_penalty": 0.3,
                "enemy_proximity_penalty": 0.0,
                "enemy_approach_penalty": 0.0,
                "boost_penalty": 0.0,
                "max_steps": 600,
                # COMPOUND: must eat AND survive — require solid food-seeking skill
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80,
                },
                "promote_window": 400,
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
                "straight_penalty": 0.0,
                "length_bonus": 0.0,
                "wall_alert_dist": 2500,
                "enemy_alert_dist": 800,
                "wall_proximity_penalty": 1.5,
                "enemy_proximity_penalty": 0.0,
                "enemy_approach_penalty": 0.0,
                "boost_penalty": 0.0,
                "starvation_penalty": 0.005,
                "starvation_grace_steps": 80,
                "starvation_max_penalty": 1.0,
                "max_steps": 500,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.10,
                "promote_window": 400,
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.95,
                "food_reward": 5.0,
                "food_shaping": 0.1,
                "survival": 0.3,
                "survival_escalation": 0.001,
                "death_wall": -40,
                "death_snake": -40,
                "straight_penalty": 0.0,
                "length_bonus": 0.0,
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 2000,
                "wall_proximity_penalty": 0.5,
                "enemy_proximity_penalty": 1.5,
                "enemy_approach_penalty": 0.5,
                "boost_penalty": 0.1,
                "starvation_penalty": 0.008,
                "starvation_grace_steps": 60,
                "starvation_max_penalty": 1.5,
                "max_steps": 2000,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500,
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
                "straight_penalty": 0.05,         
                "length_bonus": 0.02,             
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 2000,         
                "wall_proximity_penalty": 0.5,
                "enemy_proximity_penalty": 1.2,   
                "enemy_approach_penalty": 0.8,    
                "boost_penalty": 1.2,             # increased from 0.5 — make boosting expensive
                "mass_loss_penalty": 3.0,         # increased from 2.0 — losing mass must hurt
                "starvation_penalty": 0.10,       
                "starvation_grace_steps": 20,     
                "starvation_max_penalty": 3.0,
                "max_steps": 2000,
                # Promote: avg_steps >= 1000 AND avg_peak_length >= 30 (must actually grow)
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 1000,
                    "avg_peak_length": 30,
                },
                "promote_window": 500,
            },
            5: {
                "name": "MASTERY_SURVIVAL",
                "gamma": 0.95,                    
                "food_reward": 6.0,               
                "food_shaping": 0.20,             
                "survival": 0.20,                 
                "survival_escalation": 0.0002,    
                "death_wall": -45,                
                "death_snake": -50,               
                "straight_penalty": 0.0,
                "length_bonus": 0.05,             
                "wall_alert_dist": 2500,
                "enemy_alert_dist": 2500,         
                "wall_proximity_penalty": 0.35,   
                "enemy_proximity_penalty": 0.35,  
                "enemy_approach_penalty": 0.1,    
                "boost_penalty": 2.0,             # doubled from 1.0/0.4 — stop boost spam
                "mass_loss_penalty": 4.0,         # increased from 1.5/2.0 — preserve mass at all costs
                "starvation_penalty": 0.03,       
                "starvation_grace_steps": 60,     
                "starvation_max_penalty": 1.5,    
                "contest_food_reward": 0.75,      
                "enemy_zone_control_reward": 0.04,
                "kill_opportunity_reward": 12.0,  
                "max_steps": 2500,                
                # Promote: avg_steps >= 1500 AND avg_peak_length >= 80 (serious growth)
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 1500,
                    "avg_peak_length": 80,
                },
                "promote_window": 500,
            },
            6: {
                "name": "APEX_PREDATOR",
                "gamma": 0.99,
                "food_reward": 8.0,               # leave room for explicit contested/kill rewards
                "food_shaping": 0.03,             # mild shaping
                "survival": 0.10,                 # don't pay too much for passive life
                "survival_escalation": 0.0001,
                "death_wall": -40,
                "death_snake": -30,               # reduced snake death penalty — accept combat risk
                "straight_penalty": 0.0,
                "length_bonus": 0.01,             # growth matters, but pressure/kill signals should dominate
                "wall_alert_dist": 2000,
                "enemy_alert_dist": 1500,         # narrower radar — approach enemies
                "wall_proximity_penalty": 0.5,
                "enemy_proximity_penalty": 0.15,  # very low — contested space is encouraged
                "enemy_approach_penalty": 0.0,    # zero — closing in is OK
                "boost_penalty": 0.0,             # boost is free — use it to cut off enemies
                "mass_loss_penalty": 0.5,
                "starvation_penalty": 0.012,      # push to hunt, not idle
                "starvation_grace_steps": 80,
                "starvation_max_penalty": 2.5,
                "contest_food_reward": 1.0,
                "enemy_zone_control_reward": 0.06,
                "kill_opportunity_reward": 18.0,
                "max_steps": 99999,               # no limit — final stage
                "promote_metric": None,           # terminal stage
                "promote_threshold": None,
                "promote_window": 100,
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
            "death_wall": -50,
            "death_snake": -40,
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
            "food_shaping": 0.05,   # Strongly encourage seeking food
            "survival": 0.05,
            "death_wall": -50,
            "death_snake": -40,
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
