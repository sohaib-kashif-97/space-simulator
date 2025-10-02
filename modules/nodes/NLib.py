from enum import Enum


# Status enumeration for behavior tree nodes
class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

# Base class for all behavior tree nodes
class Node:
    def __init__(self, name):
        self.name = name

    async def run(self, agent, blackboard):
        raise NotImplementedError

# Sequence node: Runs child nodes in sequence until one fails
class Sequence(Node):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    async def run(self, agent, blackboard):
        for child in self.children:
            status = await child.run(agent, blackboard)
            if status == Status.RUNNING:
                continue
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS

# Fallback node: Runs child nodes in sequence until one succeeds
class Fallback(Node):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    async def run(self, agent, blackboard):
        for child in self.children:
            status = await child.run(agent, blackboard)
            if status == Status.RUNNING:
                continue
            if status != Status.FAILURE:
                return status
        return Status.FAILURE

# Parallel node: Runs all child nodes simultaneously until success or failure thresholds are met
class Parallel(Node):
    def __init__(self, name, children, success_threshold=None, failure_threshold=None):
        super().__init__(name)
        self.children = children
        self.success_threshold = success_threshold if success_threshold is not None else len(children)
        self.failure_threshold = failure_threshold if failure_threshold is not None else 1

    async def run(self, agent, blackboard):
        success_count = 0
        failure_count = 0
        for child in self.children:
            status = await child.run(agent, blackboard)
            if status == Status.SUCCESS:
                success_count += 1
                if success_count >= self.success_threshold:
                    return Status.SUCCESS
            elif status == Status.FAILURE:
                failure_count += 1
                if failure_count >= self.failure_threshold:
                    return Status.FAILURE
        return Status.RUNNING

# Synchronous action node
class SyncAction(Node):
    def __init__(self, name, action):
        super().__init__(name)
        self.action = action

    async def run(self, agent, blackboard):
        result = self.action(agent, blackboard)
        blackboard[self.name] = result
        return result

# Condition node
class Condition(Node):
    def __init__(self, name, condition_func):
        super().__init__(name)
        self.condition_func = condition_func

    async def run(self, agent, blackboard):
        result = self.condition_func(agent, blackboard)
        # Conditions should return SUCCESS or FAILURE only
        return Status.SUCCESS if result else Status.FAILURE
    
# Base Decorator node
class Decorator(Node):
    def __init__(self, name, child):
        super().__init__(name)
        self.child = child

# Inverter Decorator Node
class Inverter(Decorator):
    def __init__(self, name, child):
        super().__init__(name, child)

    async def run(self, agent, blackboard):
        status = await self.child.run(agent, blackboard)
        if status == Status.SUCCESS:
            return Status.FAILURE
        elif status == Status.FAILURE:
            return Status.SUCCESS
        return status
        