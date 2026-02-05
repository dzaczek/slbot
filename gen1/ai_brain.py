import math
import neat

class BotAgent:
    """
    Wrapper for the NEAT neural network.
    Interprets network outputs as steering commands.
    """
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.current_angle = 0  # Track current heading for smooth steering

    def decide(self, inputs, current_heading=0):
        """
        Feeds inputs to the network and returns action.
        
        inputs: List of 123 floats (ego-centric sectors + globals)
        current_heading: Current snake heading in radians
        
        Returns: (target_angle, boost)
            target_angle: Absolute angle to move towards (-PI to PI)
            boost: 0 or 1
        """
        output = self.net.activate(inputs)
        
        # Output 0: Turn direction/amount
        # Sigmoid outputs 0.0 to 1.0
        # Map to turn amount: -PI/2 to +PI/2 (max 90 degree turn per frame)
        raw_turn = output[0]
        turn_amount = (raw_turn - 0.5) * math.pi  # -PI/2 to +PI/2
        
        # Apply turn to current heading
        target_angle = current_heading + turn_amount
        
        # Normalize to -PI..PI
        while target_angle > math.pi:
            target_angle -= 2 * math.pi
        while target_angle < -math.pi:
            target_angle += 2 * math.pi
        
        # Output 1: Boost decision
        # Sigmoid outputs 0.0 to 1.0
        raw_boost = output[1]
        boost = 1 if raw_boost > 0.7 else 0
        
        return target_angle, boost
