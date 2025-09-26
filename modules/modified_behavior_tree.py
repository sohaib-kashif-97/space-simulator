import math
import random
import pygame
import importlib
from modules.utils import config

# Load additional configuration and import decision-making class dynamically
from modules.nodes.NLib import Status, Node, Sequence, Fallback, SyncAction, Condition
from plugins.my_decision_making_plugin import *

target_arrive_threshold = config['tasks']['threshold_done_by_arrival']
task_locations = config['tasks']['locations']
sampling_freq = config['simulation']['sampling_freq']
sampling_time = 1.0 / sampling_freq  # in seconds
screen_width = config['simulation']['screen_width']
screen_height = config['simulation']['screen_height']

agent_max_random_movement_duration = config.get('agents', {}).get('random_exploration_duration', None)
flocking_enabled = config.get('agents', {}).get('flocking', {}).get('enabled', False)
flocking_waypoint_duration = config['agents']['flocking']['flocking_waypoint_duration']
waypoint_transition_radius = config.get('agents', {}).get('flocking', {}).get('waypoint_transition_radius', 5)
decision_making_module_path = config['decision_making']['plugin']
boundary_margin = config.get('agents', {}).get('boundary_margin', 100)
boundary_weight = config.get('agents', {}).get('boundary_weight', 200)
module_path, class_name = decision_making_module_path.rsplit('.', 1)
decision_making_module = importlib.import_module(module_path)
decision_making_class = getattr(decision_making_module, class_name)



# BT Node List
class BehaviorTreeList:
    
    CONTROL_NODES = [        
        'Sequence',
        'Fallback'
    ]

    ACTION_NODES = [
        # 'ReturnToBaseNode',
        # 'LocalSensingNode',
        # 'DecisionMakingNode',
        # 'TaskExecutingNode', 
        # 'ExplorationNode',  
        'FlockingNode',
        'StayWithinBoundsNode',
        'NearBoundaryCondition',
        'HybridLocomotionNode'
    ]


'''
--- SIMULATOR ACTION NODES ---
'''

# Local Sensing node
class LocalSensingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._local_sensing)

    def _local_sensing(self, agent, blackboard):        
        blackboard['local_tasks_info'] = agent.get_tasks_nearby(with_completed_task = False)
        blackboard['local_agents_info'] = agent.local_message_receive()
        print(blackboard)
        return Status.SUCCESS
    
# Decision-making node
class DecisionMakingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._decide)
        self.decision_maker = decision_making_class(agent)

    def _decide(self, agent, blackboard):
        assigned_task_id = self.decision_maker.decide(blackboard)      
        agent.set_assigned_task_id(assigned_task_id)  
        blackboard['assigned_task_id'] = assigned_task_id
        if assigned_task_id is None:            
            return Status.FAILURE        
        else:                        
            return Status.SUCCESS

# Task executing node
class TaskExecutingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._execute_task)

    def _execute_task(self, agent, blackboard):        
        assigned_task_id = blackboard.get('assigned_task_id')        
        if assigned_task_id is not None:
            agent_position = agent.position
            next_waypoint = agent.tasks_info[assigned_task_id].position
            
            # Calculate norm2 distance
            distance = math.sqrt((next_waypoint[0] - agent_position[0])**2 + (next_waypoint[1] - agent_position[1])**2)
            
            assigned_task_id = blackboard.get('assigned_task_id')
            if distance < agent.tasks_info[assigned_task_id].radius + target_arrive_threshold: # Agent reached the task position                                
                if agent.tasks_info[assigned_task_id].completed:  # 이렇게 먼저 해줘야 중복해서 task_amount_done이 올라가지 않는다.                  
                    return Status.SUCCESS
                agent.tasks_info[assigned_task_id].reduce_amount(agent.work_rate)
                agent.update_task_amount_done(agent.work_rate)  # Update the amount of task done                

            # Move towards the task position
            agent.follow(next_waypoint)

        return Status.RUNNING

# Return to base node
class ReturnToBaseNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._return_to_base)
        self.return_to_base_mode = False
        self.depot_pos = pygame.Vector2(700,500)

    def _return_to_base(self, agent, blackboard):
        # Check if the assigned task is completed
        if agent.assigned_task_id is not None and agent.tasks_info[agent.assigned_task_id].completed: 
            self.return_to_base_mode = True
        
        # Move to the base if the task is completed
        if self.return_to_base_mode:            
            distance_to_base = (self.depot_pos - agent.position).length()
            if distance_to_base > (target_arrive_threshold): 
                agent.follow(self.depot_pos)
                return Status.SUCCESS
            
            self.return_to_base_mode = False
            
        # If the task is not completed, return ``FAILURE`` to allow the rest of the BT to continue
        return Status.FAILURE

# Exploration node
class ExplorationNode(SyncAction):
    def __init__(self, name, agent):
        self.random_move_time = float('inf')
        self.random_waypoint = (0, 0)
        super().__init__(name, self._random_explore)

    def _random_explore(self, agent, blackboard):
        if self.random_move_time > agent_max_random_movement_duration:
            self.random_waypoint = self.get_random_position(task_locations['x_min'], task_locations['x_max'], task_locations['y_min'], task_locations['y_max'])
            self.random_move_time = 0 
        blackboard['random_waypoint'] = self.random_waypoint        
        self.random_move_time += sampling_time   
        agent.follow(self.random_waypoint)  
        return Status.RUNNING
        
    def get_random_position(self, x_min, x_max, y_min, y_max):
        pos = (random.randint(x_min, x_max),
                random.randint(y_min, y_max))
        return pos

# Flocking node
class FlockingNode(SyncAction):

    def __init__(self, name, agent):
        self.current_waypoint = None
        self.flocking_move_time = float(0.0)  # Timer to track duration at current waypoint
        self.leader_id = 0  # Assume agent 0 is leader.
        super().__init__(name, self._flocking)


    def _flocking(self, agent, blackboard):
        
        # If Flocking is disabled in config, return FAILURE to fallback to other behaviors
        if not flocking_enabled:  
            return Status.FAILURE
        
        # Computing Center of Mass (CoM) of all agents
        blackboard['CoM'] = self.get_com(agent)

        
        # Handle shared waypoint (leader sets, others follow).
        if agent.agent_id == self.leader_id:
            if 'common_waypoint' not in blackboard:
                blackboard['common_waypoint'] = self.set_common_waypoint()
            current_shared_waypoint = blackboard['common_waypoint']
        else:
            leader = next((a for a in agent.get_all_agents() if a.agent_id == self.leader_id), None)
            if leader and 'common_waypoint' in leader.blackboard:
                current_shared_waypoint = leader.blackboard['common_waypoint']
            else:
                current_shared_waypoint = None
        
        # Detect waypoint change and reset timer for sync.
        if self.current_waypoint != current_shared_waypoint:
            self.current_waypoint = current_shared_waypoint
            self.flocking_move_time = 0.0
        
        # Leader updates the shared waypoint.
        if self.flocking_move_time > flocking_waypoint_duration:
            if agent.agent_id == self.leader_id:
                blackboard['common_waypoint'] = self.set_common_waypoint()
            self.flocking_move_time = 0.0             
               
        # If niether, perform flocking behavior and keep timer running
        self.flocking_move_time += sampling_time 
        agent.flocking(blackboard, self.current_waypoint) 
        return Status.RUNNING  # Continue flocking over ticks

    
    def set_common_waypoint(self):
        x = random.randint( 0, int(0.25 * screen_width))
        y = random.randint( 0, int(0.75 * screen_height))
        return pygame.Vector2(x, y)


    def get_com(self, agents):
        flock_agents = agents.get_all_agents()
        count = len(flock_agents)
        if count == 0:
            return pygame.Vector2(0.0, 0.0)
        sum_x, sum_y = 0.0, 0.0
        for other_agent in flock_agents:
            sum_x += other_agent.position.x
            sum_y += other_agent.position.y
        avg_x = sum_x / count
        avg_y = sum_y / count
        return pygame.Vector2(avg_x, avg_y)  

# Hybridized Locomotion node
class HybridLocomotionNode(SyncAction):
    def __init__(self, name, agent):
        self.leader_id = 0  # Assume agent 0 is leader.
        super().__init__(name, self._hybrid_locomotion) 

    def _hybrid_locomotion(self, agent, blackboard):
    
        #Get current waypoint
        if agent.agent_id == self.leader_id:
            waypoint = blackboard.get('common_waypoint')
        else:
            leader = next((a for a in agent.get_all_agents() if a.agent_id == self.leader_id), None)
            waypoint = leader.blackboard.get('common_waypoint') if leader else None

        if waypoint is None:
            return Status.FAILURE  # No waypoint to follow
        
        # Calculate distance to waypoint
        dist_to_goal = (waypoint - agent.position).length()

        #Dynamically switch between Locomotion Mechanisms based on distance
        if dist_to_goal < self.transition_radius:
            progress = dist_to_goal / self.transition_radius
            agent.separation_radius = self.min_sep_radius + (self.max_sep_radius - self.min_sep_radius) * progress  # Assume agent has separation_radius attr.
            current_speed = self.min_speed + (agent.max_speed - self.min_speed) * progress  # Assume agent.max_speed.
        else:
            agent.separation_radius = self.max_sep_radius
            current_speed = agent.max_speed

        
        
    
'''
--- SIMULATOR CONTROL NODES (WITH ACTION NODE PAIRS)---
'''

# Near Boundary Condtion Node
class NearBoundaryCondition(Condition):
    def __init__(self, name, agent):
        def condition_func(agent, blackboard):
            margin = boundary_margin
            if (agent.position.x < margin or agent.position.x > screen_width - margin or
                agent.position.y < margin or agent.position.y > screen_height - margin):
                return True
            return False
        super().__init__(name, condition_func)

# Stay within screen bounds node
class StayWithinBoundsNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._stay_within_bounds)

    def _stay_within_bounds(self, agent, blackboard):
        center_pos = pygame.Vector2(screen_width / 2, screen_height / 2)
        distance = (agent.position - center_pos).length()
        if distance > waypoint_transition_radius:
            agent.follow(center_pos, weight=boundary_weight)
        return Status.RUNNING  # Continue running to keep adjusting position.
        
      

    
    