import random

R_TYPE = {'path':-0.1, 'wall':-10, 'bomb':-30, 'destination':50}
D_REV = {'u':'d', 'd':'u', 'l':'r', 'r':'l'}

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.Ntable = {}
        self.Rtype = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = 0
        else:
            # TODO 2. Update parameters when learning
            t = sum(self.Ntable[self.sense_state()].values())
            self.epsilon = self.epsilon0 ** t

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        self.Qtable.setdefault(state, {a:.0 for a in self.valid_actions})
        self.Ntable.setdefault(state, {a:0 for a in self.valid_actions})

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():
            # TODO 5. Return whether do random choice
            return random.random() < self.epsilon
            # hint: generate a random number, and compare
            # it with epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                return random.choice(list(self.Qtable[self.state]))
            else:
                # TODO 7. Return action with highest q value
                return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        elif self.testing:
            # TODO 7. choose action with highest q value
            return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        else:
            # TODO 6. Return random choose aciton
            return random.choice(list(self.Qtable[self.state]))

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            # TODO 8. When learning, update the q table according
            # to the given rules
            state, q, n, rtype= self.state, self.Qtable, self.Ntable, self.Rtype
            # 碰墙则移除此动作，但仍然统计计数，减小随机
            if state == next_state:
                q[state].pop(action)
                n[state][action] += 1
            else:
                rtype[next_state] = r
                # 如果是炸弹根据炸弹位置对周围探索次数情况减小reward，以便探索炸弹后面的情况
                if r == R_TYPE['bomb']:
                    r *= 1 / (1 + 2.718 ** (4 - sum(n[next_state].values()) ))
                q[state][action] += self.alpha * (r + self.gamma * max(q[next_state].values()) - q[state][action])
                n[state][action] += 1                
                # 基于规则无需实践反向学习，但不能改变终点的Q值
                if state in rtype and rtype[next_state] != R_TYPE['destination']:
                    action = D_REV[action]
                    next_state, state = state, next_state
                    q[state][action] += self.alpha * (r + self.gamma * max(q[next_state].values()) - q[state][action])
                    n[state][action] += 1

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
