import math
import random
import numpy as np


class PoolAgent:
    
    def __init__(self, skill_level=0.8, strategy='balanced'):
        
        self.skill_level = max(0.1, min(1.0, skill_level))  # Clamp between 0.1 and 1.0
        self.strategy = strategy
        
        # Agent characteristics based on skill level
        self.accuracy = self.skill_level  # Accuracy when aiming
        self.power_control = self.skill_level  # Ability to control shot power
        self.error_margin = 1.0 - self.skill_level  # Error margin in shots
        
        # Strategy weights - how the agent prioritizes different shots
        self.strategy_weights = self._set_strategy_weights(strategy)
        
        # Performance tracking
        self.shots_taken = 0
        self.successful_shots = 0
        
    def _set_strategy_weights(self, strategy):
        if strategy == 'aggressive':
            return {
                'pot_ball': 1.0,       # Prioritize potting balls
                'position': 0.5,       # Medium focus on position
                'safety': 0.2,         # Low focus on safety shots
                'break': 1.0,          # Strong breaks
                'combo': 0.8           # Often attempt combination shots
            }
        elif strategy == 'defensive':
            return {
                'pot_ball': 0.7,       # Still try to pot, but not at all costs
                'position': 0.8,       # High focus on position
                'safety': 1.0,         # Prioritize safety when needed
                'break': 0.6,          # Controlled breaks
                'combo': 0.3           # Rarely attempt risky combination shots
            }
        else:
            return {
                'pot_ball': 0.8,
                'position': 0.8,
                'safety': 0.7,
                'break': 0.8,
                'combo': 0.5
            }
    
    def decide_shot(self, game_state):
        
        if game_state.get('is_break', False):
            return self._break_shot(game_state)
        
        potential_shots = self._analyze_possible_shots(game_state)
        
        if not potential_shots:
            return self._safety_shot(game_state)
        
        scored_shots = self._score_shots(potential_shots, game_state)
        
        if random.random() < self.skill_level * 0.8:

            best_shot = scored_shots[0]
        else:

            candidates = min(3, len(scored_shots))
            best_shot = random.choice(scored_shots[:candidates])

        final_shot = self._apply_error(best_shot)
        self.shots_taken += 1
        return final_shot
    
    def _break_shot(self, game_state):

        table_width = game_state['table_width']
        table_height = game_state['table_height']
        
        cue_ball = game_state['cue_ball']
        rack_x = table_width * 3 // 4
        rack_y = table_height // 2
        
        offset_y = random.uniform(-0.2, 0.2) * 2 * cue_ball['radius']
        target_y = rack_y + offset_y
        dx = rack_x - cue_ball['x']
        dy = target_y - cue_ball['y']
        angle = math.atan2(dy, dx)
        
        # Breaks are typically high power
        power = 0.85 + random.uniform(-0.05, 0.15) * self.error_margin
        
        return {
            'angle': angle,
            'power': power,
            'english': (0, 0),  # No english on break
            'shot_type': 'break'
        }
    
    def _analyze_possible_shots(self, game_state):
        """
        Analyze the table to find all possible valid shots.
        Returns a list of potential shots.
        """
        shots = []
        cue_ball = game_state['cue_ball']
        balls = game_state['balls']
        player_type = game_state['player_type']  # 'solid', 'striped', or None
        holes = game_state[ 'holes']
        
        # Determine which balls to target
        target_balls = []
        
        # If player type is not yet assigned, any ball is valid
        if player_type is None:
            for ball in balls:
                if not ball['potted'] and ball['number'] != 8:
                    target_balls.append(ball)
        
        # If player needs to hit 8-ball (all their balls are potted)
        elif game_state.get('should_hit_8_ball', False):
            for ball in balls:
                if not ball['potted'] and ball['number'] == 8:
                    target_balls = [ball]
                    break
        
        # Otherwise target only their type
        else:
            for ball in balls:
                if not ball['potted'] and ball['number'] != 8:
                    ball_type = 'striped' if ball['is_striped'] else 'solid'
                    if ball_type == player_type:
                        target_balls.append(ball)
        
        # For each target ball, check if it can be potted in any hole
        for ball in target_balls:
            for hole in holes:
                # Calculate direct shot
                direct_shot = self._evaluate_direct_shot(cue_ball, ball, hole, balls)
                if direct_shot:
                    shots.append(direct_shot)
                
                # Check for combinations (only if skill level is high enough)
                if self.skill_level > 0.6 and self.strategy_weights['combo'] > 0.3:
                    combo_shots = self._find_combination_shots(cue_ball, ball, hole, balls)
                    shots.extend(combo_shots)
        
        return shots
    
    def _evaluate_direct_shot(self, cue_ball, target_ball, hole, all_balls):
        """Evaluate if a direct shot from cue ball to target ball to hole is possible"""
        # Calculate needed angle to pot the ball
        dx_to_ball = target_ball['x'] - cue_ball['x']
        dy_to_ball = target_ball['y'] - cue_ball['y']
        distance_to_ball = math.hypot(dx_to_ball, dy_to_ball)
        
        # If balls are too close, it might be difficult to hit properly
        if distance_to_ball < cue_ball['radius'] + target_ball['radius'] + 5:
            return None
        
        # Calculate angle from target ball to hole
        dx_to_hole = hole[0] - target_ball['x']
        dy_to_hole = hole[1] - target_ball['y']
        distance_to_hole = math.hypot(dx_to_hole, dy_to_hole)
        angle_to_hole = math.atan2(dy_to_hole, dx_to_hole)
        
        # Calculate where the cue ball needs to hit the target ball
        # The cue ball should be in the opposite direction from the hole
        required_angle = (angle_to_hole + math.pi) % (2 * math.pi)
        
        # Calculate where the cue ball needs to be to make this shot
        required_x = target_ball['x'] + math.cos(required_angle) * (cue_ball['radius'] + target_ball['radius'])
        required_y = target_ball['y'] + math.sin(required_angle) * (cue_ball['radius'] + target_ball['radius'])
        
        # Calculate angle to strike the cue ball
        dx_strike = target_ball['x'] - cue_ball['x']
        dy_strike = target_ball['y'] - cue_ball['y']
        strike_angle = math.atan2(dy_strike, dx_strike)
        
        # Check if any balls are in the way of the shot
        for ball in all_balls:
            if ball['potted'] or ball == target_ball:
                continue
                
            # Check if ball is between cue ball and target ball
            if self._ball_in_path(cue_ball, target_ball, ball):
                return None
                
            # Check if ball is between target ball and hole
            if self._ball_in_path({'x': target_ball['x'], 'y': target_ball['y']}, 
                                 {'x': hole[0], 'y': hole[1]}, ball):
                return None
        
        # Calculate difficulty based on distance and angles
        difficulty = (distance_to_ball + distance_to_hole) / 1000.0
        
        # Calculate required power (simplified)
        # Longer shots need more power but we cap it
        power = min(0.3 + (distance_to_ball + distance_to_hole) / 800.0, 0.95)
        
        return {
            'angle': strike_angle,
            'power': power,
            'english': (0, 0),  # No english for basic shots
            'target_ball': target_ball,
            'hole': hole,
            'difficulty': difficulty,
            'shot_type': 'direct'
        }
    
    def _find_combination_shots(self, cue_ball, final_target, hole, all_balls):
        """Find possible combination shots (hitting another ball first)"""
        combo_shots = []
        
        # Only attempt combos with intermediate balls that aren't our target type
        intermediates = [ball for ball in all_balls if 
                        not ball['potted'] and 
                        ball != final_target]
        
        for intermediate in intermediates:
            # Check if intermediate ball can hit final target into hole
            dx = final_target['x'] - intermediate['x'] 
            dy = final_target['y'] - intermediate['y']
            dist = math.hypot(dx, dy)
            angle_to_target = math.atan2(dy, dx)
            
            # Check if intermediate can be positioned to hit target
            dx_to_hole = hole[0] - final_target['x']
            dy_to_hole = hole[1] - final_target['y']
            angle_to_hole = math.atan2(dy_to_hole, dx_to_hole)
            
            # If angles are close enough, this combo might work
            angle_diff = abs((angle_to_target - angle_to_hole + math.pi) % (2 * math.pi) - math.pi)
            if angle_diff < 0.5:  # About 30 degrees or less
                # Calculate where to hit the intermediate
                required_angle = (angle_to_target + math.pi) % (2 * math.pi)
                
                # Calculate angle from cue ball to intermediate
                dx_to_int = intermediate['x'] - cue_ball['x']
                dy_to_int = intermediate['y'] - cue_ball['y']
                angle_to_int = math.atan2(dy_to_int, dx_to_int)
                
                # Check paths for obstacles
                path_clear = True
                for ball in all_balls:
                    if ball['potted'] or ball == intermediate or ball == final_target:
                        continue
                    
                    if (self._ball_in_path(cue_ball, intermediate, ball) or
                        self._ball_in_path(intermediate, final_target, ball) or
                        self._ball_in_path(final_target, {'x': hole[0], 'y': hole[1]}, ball)):
                        path_clear = False
                        break
                
                if path_clear:
                    difficulty = 0.5 + dist / 500.0  # Combos are harder
                    power = min(0.4 + dist / 500.0, 0.9)
                    
                    combo_shots.append({
                        'angle': angle_to_int,
                        'power': power,
                        'english': (0, 0),
                        'intermediate_ball': intermediate,
                        'target_ball': final_target,
                        'hole': hole,
                        'difficulty': difficulty,
                        'shot_type': 'combo'
                    })
        
        return combo_shots
    
    def _ball_in_path(self, start, end, obstacle):
        """Check if obstacle ball is in the path between start and end points"""
        # Calculate line parameters
        line_length = math.hypot(end['x'] - start['x'], end['y'] - start['y'])
        if line_length < 1:  # Avoid division by zero
            return False
            
        # Parametric representation of the line
        dx = (end['x'] - start['x']) / line_length
        dy = (end['y'] - start['y']) / line_length
        
        # Calculate closest point on line to obstacle
        t = dx * (obstacle['x'] - start['x']) + dy * (obstacle['y'] - start['y'])
        
        # If closest point is not on line segment, obstacle is not in the way
        if t < 0 or t > line_length:
            return False
            
        # Calculate closest point coordinates
        closest_x = start['x'] + t * dx
        closest_y = start['y'] + t * dy
        
        # Check if obstacle is close enough to the line to block it
        distance = math.hypot(obstacle['x'] - closest_x, obstacle['y'] - closest_y)
        return distance < obstacle['radius'] + 5  # Add a small margin
    
    def _score_shots(self, shots, game_state):
        """Score and rank possible shots based on difficulty and strategy"""
        if not shots:
            return []
            
        scored_shots = []
        
        for shot in shots:
            # Base score - lower difficulty is better
            base_score = 1.0 - shot['difficulty']
            
            # Apply strategy weights
            if shot['shot_type'] == 'direct':
                shot_score = base_score * self.strategy_weights['pot_ball']
            elif shot['shot_type'] == 'combo':
                shot_score = base_score * self.strategy_weights['combo'] * 0.8  # Combos are riskier
            else:
                shot_score = base_score * 0.5  # Other shot types
            
            # Factor in position after the shot (simplified)
            # This would ideally predict where the cue ball ends up
            position_score = 0.5  # Default middle value
            shot_score = shot_score * 0.7 + position_score * 0.3 * self.strategy_weights['position']
            
            # Add score to the shot dictionary
            shot['score'] = shot_score
            scored_shots.append(shot)
        
        # Sort by score (highest first)
        return sorted(scored_shots, key=lambda x: x['score'], reverse=True)
    
    def _safety_shot(self, game_state):
        """
        Play a defensive safety shot when no good pot is available
        """
        cue_ball = game_state['cue_ball']
        balls = game_state['balls']
        player_type = game_state['player_type']
        
        # Find opponent's balls
        opponent_balls = []
        if player_type:
            opponent_type = 'striped' if player_type == 'solid' else 'solid'
            for ball in balls:
                if not ball['potted'] and ball['number'] != 8:
                    ball_type = 'striped' if ball['is_striped'] else 'solid'
                    if ball_type == opponent_type:
                        opponent_balls.append(ball)
        
        # If opponent's balls identified, try to snooker them
        if opponent_balls:
            # Find a ball to hit that will leave the cue ball in a difficult position
            for ball in balls:
                if not ball['potted'] and (player_type is None or 
                                          (ball['is_striped'] and player_type == 'striped') or 
                                          (not ball['is_striped'] and player_type == 'solid')):
                    # Calculate an angle that results in the cue ball stopping behind another ball
                    dx = ball['x'] - cue_ball['x']
                    dy = ball['y'] - cue_ball['y']
                    angle = math.atan2(dy, dx)
                    
                    # Use lower power for safety shots
                    safety_power = 0.3 + random.uniform(-0.1, 0.1) * self.error_margin
                    
                    return {
                        'angle': angle,
                        'power': safety_power,
                        'english': (0, -0.2),  # Slight backspin to help control
                        'shot_type': 'safety'
                    }
        
        # Default safety: hit any valid ball softly
        valid_targets = []
        for ball in balls:
            if not ball['potted'] and ball['number'] != 8:
                if player_type is None or (ball['is_striped'] and player_type == 'striped') or (not ball['is_striped'] and player_type == 'solid'):
                    valid_targets.append(ball)
        
        if valid_targets:
            target = random.choice(valid_targets)
            dx = target['x'] - cue_ball['x']
            dy = target['y'] - cue_ball['y']
            angle = math.atan2(dy, dx)
            
            return {
                'angle': angle,
                'power': 0.2 + random.uniform(-0.05, 0.05),
                'english': (0, -0.1),
                'shot_type': 'safety'
            }
        
        # If no valid target, hit with minimal power in a safe direction
        return {
            'angle': random.uniform(0, 2 * math.pi),
            'power': 0.15,
            'english': (0, 0),
            'shot_type': 'safety'
        }
    
    def _apply_error(self, shot):
        """Apply skill-based error to the shot parameters"""
        # Error increases as skill decreases
        angle_error = random.gauss(0, self.error_margin * 0.2)
        power_error = random.gauss(0, self.error_margin * 0.15)
        
        # Apply errors
        shot['angle'] = (shot['angle'] + angle_error) % (2 * math.pi)
        shot['power'] = max(0.1, min(1.0, shot['power'] + power_error))
        
        return shot
    
    def update_performance(self, was_successful):
        """Update agent's performance tracking after a shot"""
        if was_successful:
            self.successful_shots += 1

    def get_performance_stats(self):
        """Return the agent's performance statistics"""
        if self.shots_taken == 0:
            success_rate = 0
        else:
            success_rate = self.successful_shots / self.shots_taken
            
        return {
            'shots_taken': self.shots_taken,
            'successful_shots': self.successful_shots,
            'success_rate': success_rate,
            'skill_level': self.skill_level,
            'strategy': self.strategy
        }


class PoolAgentFactory:
    
    @staticmethod
    def create_agent(agent_type='balanced', skill_level=None):
        
        if agent_type == 'aggressive':
            agent_skill = 0.75 if skill_level is None else skill_level
            return PoolAgent(skill_level=agent_skill, strategy='aggressive')
            
        elif agent_type == 'defensive':
            agent_skill = 0.8 if skill_level is None else skill_level
            return PoolAgent(skill_level=agent_skill, strategy='defensive')
            
        elif agent_type == 'beginner':
            agent_skill = 0.3 if skill_level is None else skill_level
            # Beginners tend to be more aggressive and less strategic
            return PoolAgent(skill_level=agent_skill, strategy='aggressive')
            
        elif agent_type == 'expert':
            agent_skill = 0.95 if skill_level is None else skill_level
            return PoolAgent(skill_level=agent_skill, strategy='balanced')
            
        elif agent_type == 'random':
            # Random agent ignores strategy and just makes random decisions
            class RandomPoolAgent(PoolAgent):
                def decide_shot(self, game_state):
                    """Make a completely random shot"""
                    cue_ball = game_state['cue_ball']
                    
                    # Random angle
                    angle = random.uniform(0, 2 * math.pi)
                    
                    # Random power
                    power = random.uniform(0.2, 1.0)
                    
                    return {
                        'angle': angle,
                        'power': power,
                        'english': (0, 0),
                        'shot_type': 'random'
                    }
            
            return RandomPoolAgent(skill_level=0.1, strategy='balanced')
            
        else:  # balanced (default)
            agent_skill = 0.7 if skill_level is None else skill_level
            return PoolAgent(skill_level=agent_skill, strategy='balanced')


# Function to convert the game state for the agent
def convert_game_state_for_agent(game):
    """
    Convert the game state from the Pool Game class to a format
    the agent can understand and use for decision making.
    """
    state = {
        'table_width': game.WIDTH,
        'table_height': game.HEIGHT,
        'cue_ball': {
            'x': game.cue_ball.x,
            'y': game.cue_ball.y,
            'radius': game.cue_ball.radius,
            'vx': game.cue_ball.vx,
            'vy': game.cue_ball.vy,
            'potted': game.cue_ball.potted
        },
        'balls': [
            {
                'x': ball.x,
                'y': ball.y,
                'radius': ball.radius,
                'number': ball.number,
                'is_striped': ball.is_striped,
                'potted': ball.potted,
                'vx': ball.vx,
                'vy': ball.vy
            } for ball in game.balls
        ],
        'holes': game.holes,
        'player_type': game.players[game.current_player]['type'],
        'current_player': game.current_player,
        'is_break': len([ball for ball in game.balls if not ball.potted]) == len(game.balls),
        'should_hit_8_ball': game.should_hit_8_ball()
    }
    
    return state


# Example usage
if __name__ == "__main__":
    # Create different types of agents
    balanced_agent = PoolAgentFactory.create_agent('balanced')
    aggressive_agent = PoolAgentFactory.create_agent('aggressive')
    defensive_agent = PoolAgentFactory.create_agent('defensive')
    expert_agent = PoolAgentFactory.create_agent('expert')
    beginner_agent = PoolAgentFactory.create_agent('beginner')
    
    # Example game state (would normally come from the game)
    example_state = {
        'table_width': 1200,
        'table_height': 800,
        'cue_ball': {'x': 300, 'y': 400, 'radius': 12, 'vx': 0, 'vy': 0, 'potted': False},
        'balls': [
            {'x': 600, 'y': 400, 'radius': 12, 'number': 1, 'is_striped': False, 'potted': False, 'vx': 0, 'vy': 0},
            {'x': 650, 'y': 420, 'radius': 12, 'number': 8, 'is_striped': False, 'potted': False, 'vx': 0, 'vy': 0},
            {'x': 700, 'y': 380, 'radius': 12, 'number': 9, 'is_striped': True, 'potted': False, 'vx': 0, 'vy': 0}
        ],
        'holes': [(80, 80), (600, 70), (1120, 80), (80, 720), (600, 730), (1120, 720)],
        'player_type': 'solid',
        'current_player': 0,
        'is_break': False,
        'should_hit_8_ball': False
    }
    
    # Get decisions from different agents
    print("Balanced agent's decision:", balanced_agent.decide_shot(example_state))
    print("Aggressive agent's decision:", aggressive_agent.decide_shot(example_state))
    print("Defensive agent's decision:", defensive_agent.decide_shot(example_state))
    print("Expert agent's decision:", expert_agent.decide_shot(example_state))
    print("Beginner agent's decision:", beginner_agent.decide_shot(example_state))