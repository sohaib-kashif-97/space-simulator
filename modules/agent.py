import pygame
import math
import copy
from modules.modified_behavior_tree import *            #NOTE: ADJUSTED TO MODIFIED BT
from modules.utils import config, generate_positions, parse_behavior_tree
from modules.task import task_colors



# Load simulation settings
font = pygame.font.Font(None, 15)
sampling_freq = config['simulation']['sampling_freq']
sampling_time = 1.0 / sampling_freq  # in seconds
screen_width = config['simulation']['screen_width']
screen_height = config['simulation']['screen_height']

# Load agent configuration
agent_max_speed = config['agents']['max_speed']
agent_max_accel = config['agents']['max_accel']
max_angular_speed = config['agents']['max_angular_speed']
agent_approaching_to_target_radius = config['agents']['target_approaching_radius']
agent_track_size = config['simulation']['agent_track_size']
work_rate = config['agents']['work_rate']
agent_communication_radius = config['agents']['communication_radius']
agent_situation_awareness_radius = config.get('agents', {}).get('situation_awareness_radius', 0)
flocking_condition = config.get('agents', {}).get('flocking', {}).get('enabled', False)
sep_weight = config['agents']['flocking']['separation_weight'] if flocking_condition else 0
aln_weight = config['agents']['flocking']['alignment_weight'] if flocking_condition else 0
chn_weight = config['agents']['flocking']['cohesion_weight'] if flocking_condition else 0
sep_radius = config['agents']['flocking']['separation_radius'] if flocking_condition else 0
max_flocking_speed = config['agents']['flocking']['max_flocking_speed'] if flocking_condition else 0
max_flocking_accel = config['agents']['flocking']['max_flocking_accel'] if flocking_condition else 0
waypoint_transition_radius = config['agents']['flocking']['waypoint_transition_radius'] if flocking_condition else 0

# Load behavior tree
behavior_tree_xml = config['agents']['behavior_tree_xml']
xml_root = parse_behavior_tree(f"bt_xml/{behavior_tree_xml}")


def generate_agents(tasks_info):
    agent_quantity = config['agents']['quantity']
    agent_locations = config['agents']['locations']

    agents_positions = generate_positions(agent_quantity,
                                      agent_locations['x_min'],
                                      agent_locations['x_max'],
                                      agent_locations['y_min'],
                                      agent_locations['y_max'],
                                      radius=agent_locations['non_overlap_radius'])

    # Initialize agents
    agents = [Agent(idx, pos, tasks_info) for idx, pos in enumerate(agents_positions)]

    # Provide the global info and create behavior tree
    for agent in agents:
        agent.set_global_info_agents(agents)
        agent.create_behavior_tree()

    return agents



class Agent:
    def __init__(self, agent_id, position, tasks_info):
        self.agent_id = agent_id
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.max_speed = agent_max_speed
        self.max_accel = agent_max_accel
        self.max_angular_speed = max_angular_speed
        self.work_rate = work_rate
        self.memory_location = []                                           # To draw track
        self.rotation = 0                                                   # Initial rotation
        self.color = (0, 0, 255)                                            # Blue color
        self.blackboard = {}                                                # Blackboard --> { BT Node: Comprises of an Agent's Local info including messages, tasks, etc}.

        self.tasks_info = tasks_info                                        # global info
        self.agents_info = None                                             # global info
        self.communication_radius = agent_communication_radius
        self.situation_awareness_radius = agent_situation_awareness_radius
        self.agents_nearby = []
        self.message_to_share = {}
        self.messages_received = []
        self.assigned_task_id = None                                        # Local decision-making result --> ID
        self.planned_tasks = []                                             # Local decision-making result --> Task objects for visualization
        
        self.distance_moved = 0.0                                           # Recorded Evaluation Metrics
        self.task_amount_done = 0.0      

        # Flocking Variables
        self.current_flocking_waypoint = None
        self.sep_radius = sep_radius
        self.sep_weight = sep_weight
        self.aln_weight = aln_weight
        self.chn_weight = chn_weight
        self.waypoint_transition_radius = waypoint_transition_radius
        self.max_flocking_speed = max_flocking_speed
        self.max_flocking_accel = max_flocking_accel



    '''
    Methods interacting with Agent's Behavior Tree
    '''
    async def run_tree(self):
        #Asynchronoursly reset and then run BT (parse tree from xml file)
        self._reset_bt_action_node_status()
        return await self.tree.run(self, self.blackboard)       # Trigger the behavior tree execution?


    def _reset_bt_action_node_status(self):
        action_nodes = BehaviorTreeList.ACTION_NODES
        self.blackboard = {key: None if key in action_nodes else value for key, value in self.blackboard.items()}


    def create_behavior_tree(self):
        self.tree = self._create_behavior_tree()


    def _create_behavior_tree(self):
        behavior_tree = self._parse_xml_to_bt(xml_root.find('BehaviorTree'))
        return behavior_tree        


    def _parse_xml_to_bt(self, xml_node):
        node_type = xml_node.tag
        children = []

        for child in xml_node:
            children.append(self._parse_xml_to_bt(child))

        if node_type in BehaviorTreeList.CONTROL_NODES:
            control_class = globals()[node_type]  # Control class should be globally available
            return control_class(node_type, children=children)
        elif node_type in BehaviorTreeList.ACTION_NODES:
            action_class = globals()[node_type]  # Action class should be globally available
            return action_class(node_type, self)
        elif node_type == "BehaviorTree": # Root
            return children[0]
        else:
            raise ValueError(f"[ERROR] Unknown behavior node type: {node_type}")    



    '''
    Methods for Agent's Communication (Mechanisms to recieve messages from nearby agents)
    '''
    def reset_messages_received(self):
        self.messages_received = []


    def local_message_receive(self):
        self.agents_nearby = self.get_agents_nearby()
        for other_agent in self.agents_nearby:
            if other_agent.agent_id != self.agent_id:                         
                self.receive_message(other_agent.message_to_share)
        return self.agents_nearby


    def receive_message(self, message):
        self.messages_received.append(message)  



    '''
    Methods for Agent's Flocking Behavior
    '''
    def flocking(self, agent, blackboard):
        
        # Retrieve Agent's current position from the blackboard
        locomotion_vel = self.locomotion_rule()
        cohesion_vel = self.cohesion_rule()
        alignment_vel = self.alignment_rule()
        separation_vel = self.separation_rule()

        # locomotion_vel = pygame.Vector2(0.0, 0.0)
        # cohesion_vel = pygame.Vector2(0.0, 0.0)
        # alignment_vel = pygame.Vector2(0.0, 0.0)
        # separation_vel = pygame.Vector2(0.0, 0.0)

        net_agent_vel = pygame.Vector2(
            locomotion_vel[0] + cohesion_vel[0] + alignment_vel[0] + separation_vel[0],
            locomotion_vel[1] + cohesion_vel[1] + alignment_vel[1] + separation_vel[1]
        )
        if net_agent_vel.length() > 0:
            net_agent_vel.normalize_ip()
            net_agent_vel *= self.max_flocking_speed  
        
        self.acceleration = (net_agent_vel - self.velocity) / sampling_time
        self.acceleration = self.limit(self.acceleration, self.max_flocking_accel)
        

    def locomotion_rule(self):
        
        # # Initializing four corner of the screen as waypoints set randomly
        # waypoints = [pygame.Vector2( (screen_width / 5) , (screen_height / 5)),
        #             pygame.Vector2( 4 * (screen_height / 5), (screen_width / 5) ),
        #             pygame.Vector2( (screen_width / 5) , 4 * (screen_height / 5)),
        #             pygame.Vector2( 4 * (screen_height / 5) , 4 * (screen_height / 5))]
        
        # Persist waypoint; change only when close or none set
        # if self.current_flocking_waypoint is None or (self.position - self.current_flocking_waypoint).length() < self.waypoint_transition_radius:
        #     ridx = random.randint(0, len(waypoints) - 1)
        #     self.current_flocking_waypoint = waypoints[ridx]

        if self.current_flocking_waypoint is None:
            self.current_flocking_waypoint = pygame.Vector2( (screen_width / 5) , (screen_height / 5))
        else:
            self.current_flocking_waypoint = self.current_flocking_waypoint

        locomotion_vector = self.current_flocking_waypoint - self.position
        if locomotion_vector.length() > 0:
            locomotion_vector.normalize_ip()
            locomotion_vector *= self.max_flocking_speed

        return (locomotion_vector.x, locomotion_vector.y)
    

    def cohesion_rule(self):
        
        # Initialzing variables relative to Cohesion Rule
        sum_x, sum_y = 0.0, 0.0
        center_x, center_y = 0.0, 0.0
        agents_flocking_info = self.get_all_agents()    
        count = len(agents_flocking_info)  
        
        # If there are no agents in the simulation, skip onwards...
        if count == 0:
            return pygame.Vector2(0.0, 0.0)
        
        #Compute Center of Mass (CoM) w.r.t all the other agents
        for other_agent in agents_flocking_info:
            if other_agent.agent_id != self.agent_id:
                sum_x += other_agent.position.x
                sum_y += other_agent.position.y
        center_x = sum_x / count
        center_y = sum_y / count
        
        # Derive Unit Vector to move agents to their CoM
        cohesion_vector = pygame.Vector2(center_x, center_y) - self.position
        if cohesion_vector.length() > 0:
            cohesion_vector.normalize_ip()

        cohesion_vector = cohesion_vector * self.chn_weight
        return (cohesion_vector.x, cohesion_vector.y)
    

    def alignment_rule(self):

        # Initialzing variables relative to Cohesion Rule
        sum_vx, sum_vy = 0.0, 0.0
        avg_vx, avg_vy = 0.0, 0.0
        agents_flocking_info = self.get_all_agents()    
        count = len(agents_flocking_info)

        # If there are no agents in the simulation, skip onwards...
        if count == 0:
            return pygame.Vector2(0.0, 0.0)
        
        #Compute Center of Mass (CoM) w.r.t all the other agents
        for other_agent in agents_flocking_info:
            sum_vx += other_agent.velocity.x
            sum_vy += other_agent.velocity.y
        avg_vx = sum_vx / count
        avg_vy = sum_vy / count

        alignment_vector = pygame.Vector2(avg_vx, avg_vy)
        dist = alignment_vector.length()

        #Computing the Magnitude of the Average Velocity
        if dist > 0:
            alignment_vector = alignment_vector / dist

        alignment_vector = alignment_vector * self.aln_weight
        return (alignment_vector.x, alignment_vector.y)
    

    def separation_rule(self):
        
        # Initializing variables relative to Separation Rule
        separation_vector = pygame.Vector2(0, 0)
        local_agents_info = self.get_agents_nearby(self.sep_radius)
        total_nearby_agents = len(local_agents_info)
        count = 0
        
        # Comparing all boids with one another to check the separation criteria
        for other_agent in local_agents_info:
            
            # Compute Vector Components away from the other boid
            other_pos = other_agent.position
            diff = self.position - other_pos
            dist = diff.length()

            # If criteria has met, compute the heading vector
            if 0 < dist <= self.sep_radius:
                diff /= dist  # Weight by distance
                separation_vector += diff
                count += 1
        
        if count > 0:
            vx = self.sep_weight * separation_vector.x * (1 + ((count + 1) / total_nearby_agents))
            vy = self.sep_weight * separation_vector.y * (1 + ((count + 1) / total_nearby_agents))
        else:
            vx = self.sep_weight * separation_vector.x
            vy = self.sep_weight * separation_vector.y

        return (vx, vy)
    

    
    '''
    Methods for Agent's Kinematics
    '''
    def update(self):
        # Update velocity and position
        self.velocity += self.acceleration * sampling_time
        self.velocity = self.limit(self.velocity, self.max_speed)
        self.position += self.velocity * sampling_time
        self.acceleration *= 0  # Reset acceleration

        # Calculate the distance moved in this update and add to distance_moved
        self.distance_moved += self.velocity.length() * sampling_time
        
        # Memory of positions to draw track
        self.memory_location.append((self.position.x, self.position.y))
        if len(self.memory_location) > agent_track_size:
            self.memory_location.pop(0)

        # Update rotation
        desired_rotation = math.atan2(self.velocity.y, self.velocity.x)
        rotation_diff = desired_rotation - self.rotation
        while rotation_diff > math.pi:
            rotation_diff -= 2 * math.pi
        while rotation_diff < -math.pi:
            rotation_diff += 2 * math.pi

        # Limit angular velocity
        if abs(rotation_diff) > self.max_angular_speed:
            rotation_diff = math.copysign(self.max_angular_speed, rotation_diff)
        self.rotation += rotation_diff * sampling_time


    def reset_movement(self):
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)


    # MODIFY: Detect if the agent has a boundary_weight attribute attribute passed from modified_bt module. If so, apply the boundary steering force as well... 
    def follow(self, target, weight=1.0):
        # Calculate locomotion_vector velocity
        locomotion_vector = target - self.position
        d = locomotion_vector.length()

        if d < agent_approaching_to_target_radius:
            # Apply arrival behavior
            locomotion_vector.normalize_ip()
            locomotion_vector *= self.max_speed * (d / agent_approaching_to_target_radius)  # Adjust speed based on distance
        else:
            if weight != 1.0:
                locomotion_vector.normalize_ip()
                locomotion_vector *= self.max_speed
                locomotion_vector *= weight   # Apply the weight for boundary steering force
            else:
                locomotion_vector.normalize_ip()
                locomotion_vector *= self.max_speed

        steer = locomotion_vector - self.velocity
        steer = self.limit(steer, self.max_accel)
        self.applyForce(steer)


    def limit(self, vector, max_value):
        if vector.length_squared() > max_value**2:
            vector.scale_to_length(max_value)
        return vector


    def applyForce(self, force):
        self.acceleration += force



    '''
    Methods for Agent's Visualization
    '''
    def draw(self, screen):
        size = 10
        angle = self.rotation

        # Calculate the triangle points based on the current position and angle
        p1 = pygame.Vector2(self.position.x + size * math.cos(angle), self.position.y + size * math.sin(angle))
        p2 = pygame.Vector2(self.position.x + size * math.cos(angle + 2.5), self.position.y + size * math.sin(angle + 2.5))
        p3 = pygame.Vector2(self.position.x + size * math.cos(angle - 2.5), self.position.y + size * math.sin(angle - 2.5))

        self.update_color()
        pygame.draw.polygon(screen, self.color, [p1, p2, p3])


    def draw_tail(self, screen):
        # Draw track
        if len(self.memory_location) >= 2:
            pygame.draw.lines(screen, self.color, False, self.memory_location, 1)               
        

    def draw_communication_topology(self, screen, agents):
        # Draw lines to neighbor agents
        for neighbor_agent in self.agents_nearby:
            if neighbor_agent.agent_id > self.agent_id:
                neighbor_position = agents[neighbor_agent.agent_id].position
                pygame.draw.line(screen, (200, 200, 200), (int(self.position.x), int(self.position.y)), (int(neighbor_position.x), int(neighbor_position.y)))


    def draw_agent_id(self, screen):
        # Draw assigned_task_id next to agent position
        text_surface = font.render(f"agent_id: {self.agent_id}", True, (50, 50, 50))
        screen.blit(text_surface, (self.position[0] + 10, self.position[1] - 10))


    def draw_assigned_task_id(self, screen):
        # Draw assigned_task_id next to agent position
        if len(self.planned_tasks) > 0:
            assigned_task_id_list = [task.task_id for task in self.planned_tasks]
        else:
            assigned_task_id_list = self.assigned_task_id
        text_surface = font.render(f"task_id: {assigned_task_id_list}", True, (50, 50, 50))
        screen.blit(text_surface, (self.position[0] + 10, self.position[1]))


    def draw_work_done(self, screen):
        # Draw assigned_task_id next to agent position
        text_surface = font.render(f"dist: {self.distance_moved:.1f}", True, (50, 50, 50))
        screen.blit(text_surface, (self.position[0] + 10, self.position[1] + 10))
        text_surface = font.render(f"work: {self.task_amount_done:.1f}", True, (50, 50, 50))
        screen.blit(text_surface, (self.position[0] + 10, self.position[1] + 20))


    def draw_situation_awareness_circle(self, screen):
        # Draw the situation awareness radius circle    
        if self.situation_awareness_radius > 0:    
            pygame.draw.circle(screen, self.color, (self.position[0], self.position[1]), self.situation_awareness_radius, 1)


    def draw_path_to_assigned_tasks(self, screen):
        # Starting position is the agent's current position
        start_pos = self.position

        # Define line thickness
        line_thickness = 3  # Set the locomotion_vector thickness for the lines        
        # line_thickness = 16-4*self.agent_id  # Set the locomotion_vector thickness for the lines        

        # For Debug
        color_list = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203) # Pink
        ]
                
        # Iterate over the assigned tasks and draw lines connecting them
        for task in self.planned_tasks:
            task_position = task.position
            pygame.draw.line(
                screen,
                # (255, 0, 0),  # Color for the path line (Red)
                color_list[self.agent_id%len(color_list)], 
                (int(start_pos.x), int(start_pos.y)),
                (int(task_position.x), int(task_position.y)),
                line_thickness  # Thickness of the line
            )
            # Update the start position for the next segment
            start_pos = task_position


    def update_color(self):        
        self.color = task_colors.get(self.assigned_task_id, (20, 20, 20))  # Default to Dark Grey if no task is assigned



    '''
    Methods for Agent's Interaction
    '''
    def set_assigned_task_id(self, task_id):
        self.assigned_task_id = task_id


    def set_planned_tasks(self, task_list): # This is for visualisation
        self.planned_tasks = task_list    


    def set_global_info_agents(self, agents_info):
        self.agents_info = agents_info


    def update_task_amount_done(self, amount):
        self.task_amount_done += amount


    def get_agents_nearby(self, radius = None):
        _communication_radius = self.communication_radius if radius is None else radius        
        if _communication_radius > 0:
            communication_radius_squared = _communication_radius ** 2        
            local_agents_info = [
                other_agent
                for other_agent in self.agents_info
                if (self.position - other_agent.position).length_squared() <= communication_radius_squared and other_agent.agent_id !=self.agent_id
            ]
        else:
            local_agents_info = self.agents_info
        return local_agents_info
    

    def get_all_agents(self):
        agents_flocking_info = [
            other_agent
            for other_agent in self.agents_info if other_agent.agent_id !=self.agent_id
        ]
        return agents_flocking_info

   
    def get_tasks_nearby(self, radius = None, with_completed_task = True):
        _situation_awareness_radius = self.situation_awareness_radius if radius is None else radius
        if _situation_awareness_radius > 0:
            situation_awareness_radius_squared = _situation_awareness_radius ** 2
            if with_completed_task: # Default
                local_tasks_info = [
                    task 
                    for task in self.tasks_info 
                    if (self.position - task.position).length_squared() <= situation_awareness_radius_squared
                ]                
            else:
                local_tasks_info = [
                    task 
                    for task in self.tasks_info 
                    if not task.completed and (self.position - task.position).length_squared() <= situation_awareness_radius_squared
                ]                                
        else:
            if with_completed_task: # Default
                local_tasks_info = self.tasks_info
            else:
                local_tasks_info = [
                    task 
                    for task in self.tasks_info 
                    if not task.completed
                ]                                                
        
        return local_tasks_info 