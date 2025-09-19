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

agent_max_random_movement_duration = config.get('agents', {}).get('random_exploration_duration', None)
flocking_condition = config.get('agents', {}).get('flocking', {}).get('enabled', False)
decision_making_module_path = config['decision_making']['plugin']
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
        'ReturnToBaseNode',
        # 'LocalSensingNode',
        # 'DecisionMakingNode',
        # 'TaskExecutingNode',  
        'FlockingNode',
        'StayWithinBoundsNode',
        # 'ExplorationNode', 
        'NearBoundaryCondition',
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

# Flocking behaviour node
class FlockingNode(SyncAction):

    def __init__(self, name, agent):
        super().__init__(name, self._flocking)

    #NOTE: Flocking Control Loop is applied per agent as this is an Action Node of the current BT 
    def _flocking(self, agent, blackboard):
        if not flocking_condition:
            return Status.FAILURE
        else:
            # Flocking Behavior Implementation
            agent.flocking(agent, blackboard)
            return Status.SUCCESS
        

'''
--- SIMULATOR CONTROL NODES (WITH ACTION NODE PAIRS)---
'''

# Near Boundary Condtion Node
class NearBoundaryCondition(Condition):
    
    def __init__(self, name, agent):
        self.boundary_margin = 100
        self.boundary_weight = 150  # Not used in condition, but consistent with StayWithinBoundsNode
        self.screen_width = config['simulation']['screen_width']
        self.screen_height = config['simulation']['screen_height']
        self.x_min = task_locations['x_min'] + self.boundary_margin
        self.x_max = task_locations['x_max'] - self.boundary_margin
        self.y_min = task_locations['y_min'] + self.boundary_margin
        self.y_max = task_locations['y_max'] - self.boundary_margin
        condition_func = lambda agent, blackboard: not (
            self.x_min <= agent.position.x <= self.x_max and 
            self.y_min <= agent.position.y <= self.y_max
        )
        super().__init__(name, condition_func)

# Stay within screen bounds node
class StayWithinBoundsNode(SyncAction):
    def __init__(self, name, agent):
        self.boundary_margin = 100        # Margin to start steering back
        self.boundary_weight = 150        # Adjustable Weight for steering back force
        self.boundary_avoidance_mode = False
        self.screen_width = config['simulation']['screen_width']
        self.screen_height = config['simulation']['screen_height']
        self.x_min = task_locations['x_min'] + self.boundary_margin
        self.x_max = task_locations['x_max'] - self.boundary_margin
        self.y_min = task_locations['y_min'] + self.boundary_margin
        self.y_max = task_locations['y_max'] - self.boundary_margin
        super().__init__(name, self._stay_within_bounds)
        
        

    def _stay_within_bounds(self, agent, blackboard):
        
        # Move the agent back within bounds until it reaches the threshold
        new_x = self.screen_width / 2
        new_y = self.screen_height / 2
        center_pos = (new_x, new_y)
        distance_to_center = (center_pos - agent.position).length()
        if distance_to_center > target_arrive_threshold: 
            agent.follow((new_x, new_y), weight=self.boundary_weight)  # Apply the boundary weight
            return Status.RUNNING
        
        # If the task is not completed, return ``FAILURE`` to allow the rest of the BT to continue
        return Status.FAILURE