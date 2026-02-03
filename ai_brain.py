import math
import neat

class BotAgent:
    """
    Wrapper for the NEAT neural network.
    """
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def decide(self, inputs):
        """
        Feeds inputs to the network and returns action.
        inputs: List of 145 floats.
        Returns: (angle, boost)
        """
        output = self.net.activate(inputs)
        
        # Output 0: Angle change
        # Network outputs usually 0..1 or -1..1 depending on config.
        # Our config says 'sigmoid' activation, so 0.0 to 1.0.
        # We map 0.0-1.0 to -PI to PI
        raw_angle = output[0]
        angle = (raw_angle * 2 * math.pi) - math.pi 
        
        # Output 1: Boost decision
        # 0.0 to 1.0
        raw_boost = output[1]
        boost = 1 if raw_boost > 0.8 else 0 # High threshold to conserve mass
        
        return angle, boost
