from itertools import combinations,permutations
import numpy as np
import multiprocessing as mp


class Tictactoe:
#     can play the game using non GUI , just run the play() method
    def __init__(self):
        
        self.states = self.get_states()
        self.wins = self.get_wins()
        self.actions = dict(enumerate(map(chr,range(ord('a'),ord('i')+1)))) # this felt good 
        self.Q = np.zeros((len(self.states),9) , np.int)
        self.training_done = mp.Queue()
        
    @staticmethod    
    def get_states():
    #     there has to be atleast 1 0 ie 1 empty space in the states as all states filled is end game which has no next state,
    #     the states have to encode the agents next move so where, either there is a single 1 meaning
    #     that the agents turn is next or the number of agent plays (2's) cannot be even as that means it is the other players turn

        comb = list(set(combinations(np.repeat( [0,1,2] , 9),9)))
        valid_configs = []
        for com in comb:
            vals, counts = np.unique(com,return_counts=True)
    #         print(vals, counts)
            if (0 in vals):
                if len(vals) == 2:
                    if counts[1] == 1 and vals[1] != 2:
                        valid_configs.append(com)

                elif len(vals) > 2:
                    if  (counts[2]==counts[1]-1) or (counts[1]==counts[2]) : 
                         valid_configs.append(com)
                else:
                     valid_configs.append(com)


        #         print(com,vals , counts)
        valid_states = []
        for config in valid_configs:
            valid_states.append(list(set(permutations(config))))

        flat_list = [item for sublist in valid_states for item in sublist]
        state_dict = dict()
        for i,state in enumerate(sorted(flat_list)):
            state_dict[i] = state
        return state_dict
    
    @staticmethod
    def check(test,array):
        return [True for win in array if set(win).issubset(test)]
    
    @staticmethod
    def get_wins():
        grid  = np.array(range(9)).reshape(3,3)
        wins = []
        wins  = [grid[i] for i in range(3)]
        wins  +=  [grid[:,i] for i in range(3)]
        wins  += [grid.diagonal()]
        wins  += [np.fliplr(grid).diagonal()]
        return wins
    
    def run_episode(self,episode_length,gamma,alpha ,epsilon ,win_reward, lose_reward, draw_reward,play_reward,queue_for_q):
    
        Q = np.zeros((len(self.states),9) , np.int)
        states = self.states
        
        for i in range(episode_length):
    #         simulate random states
            current_state_selector = np.random.randint(0,i%len(states)+1)  
            current_state = np.array(states[current_state_selector])
    #         get playable blocks
            available_blocks = np.where(np.array(current_state) == 0)[0]
    #         epsilon greedy methond ,explore , exploit
            if  np.random.random() > epsilon:   

                avialable_q_vals =   np.array([Q[current_state_selector][available_blocks],available_blocks])
#                 print(avialable_q_vals[0],"q")
#                 print(avialable_q_vals[1],"a")
                if len(np.unique(avialable_q_vals[0])) == 1:
                    action_to_take_agent = np.random.choice(available_blocks)

                else:
                 
                    action_to_take_agent = avialable_q_vals[1,np.argmax(avialable_q_vals[0])]

            else:

                action_to_take_agent = np.random.choice(available_blocks) # index/pos at which to play

            action_taken_state = current_state.copy()
            action_taken_state[action_to_take_agent] = 2

        #     did i win ?
            plays = np.where(np.array(action_taken_state) == 2)[0]
            reward = play_reward
            terminal_state = False

            if self.check(plays,self.wins):# win
                reward = win_reward
                terminal_state = True

            elif 0 not in action_taken_state:# draw
                reward = draw_reward
                terminal_state = True
            else:
    #         rival play

                posible_rival_actions  = np.where(np.array(action_taken_state) == 0)[0]# get possible plays
                action_to_take_rival = np.random.choice(posible_rival_actions) # index/pos at which to play
                action_taken_state[action_to_take_rival] = 1

                plays = np.where(np.array(action_taken_state) == 1)[0]
                if self.check(plays,self.wins):
                    reward = lose_reward
                    terminal_state = True
                elif (0 not in action_taken_state):
                    reward = draw_reward
                    terminal_state = True

                else:
                    pass

            if terminal_state:
                TD = reward
            else:    
                next_state_locator = list(states.values()).index(tuple(action_taken_state))
                TD  = reward + gamma *  Q[next_state_locator][np.argmax(Q[next_state_locator])] - Q[current_state_selector][action_to_take_agent]

            Q[current_state_selector][action_to_take_agent] +=  alpha * TD

        else:
            queue_for_q.put(Q)


    def train(self,episode_length = 5000,nprocs = None,gamma= 0.6,alpha = 0.99,epsilon = 0.4 , win_reward = 100 , lose_reward = -100 , draw_reward = -50 , play_reward = -1):        
    
        sessions = []
        queue_for_q = mp.Queue()
        if not nprocs:
            nprocs = mp.cpu_count()
        
        
#         create multiple processes to run the training episodes in paralelle

        for i in range(nprocs):
            p = mp.Process(target=self.run_episode,args=(episode_length,gamma,alpha,epsilon ,win_reward, lose_reward, draw_reward,play_reward,queue_for_q))
            p.start()
            sessions.append(p)

#         get the Q values from the Queue lol
        for i in range(nprocs):
            self.Q += queue_for_q.get()

#         wait for finish
        for session in sessions:
            session.join()
        
#         average over all that we have learned , maybe not best method ? 
        self.Q = self.Q//nprocs
        self.training_done.put(1)
       
    
    @staticmethod
    def show_game_state(state):
        
        available_blocks = np.where(np.array(state) == 0 )[0]
        state = np.array(state).reshape(3,3)

        print('a|b|c',state[0])
        print('d|e|f',state[1])
        print('g|h|i',state[2])
        return available_blocks
    
    def play(self):# manual play
        game_state = np.zeros(9,int)
        blocks = {j:i for i,j in self.actions.items()}   
        while 1:
#             GAME STATE
#             ==================================================================
            availble_blocks = self.show_game_state(game_state)
            availble_blocks = [self.actions[i] for i in availble_blocks]
            print('choose block to play in:',availble_blocks)
#             ==================================================================
#             HUMAN PLAY
            human_action = input('enter block to play in >')

            if human_action not in availble_blocks:
                print('invalid')
                continue
            else:
                human_action = blocks[human_action] # convert letter to index
                game_state[human_action] = 1

                plays = np.where(np.array(game_state) == 1)[0]

                if self.check(plays,self.wins):
                    print('HUMAN WINS')
                    self.show_game_state(game_state)
                    break
                elif 0 not in game_state:
                    print('DRAW')
                    break
                    
#                 ================================================================================
#                 AGENT PLAY
                what_state = list(self.states.values()).index(tuple(game_state))
                available_blocks = np.where(np.array(game_state) == 0)[0]
                avialable_q_vals =   np.array([self.Q[what_state][available_blocks],available_blocks])

                if len(np.unique(avialable_q_vals[0])) == 1:
                    agent_action = np.random.choice(available_blocks)
                else:
                    agent_action = avialable_q_vals[1,np.argmax(avialable_q_vals[0])]

                game_state[agent_action] = 2

                plays = np.where(np.array(game_state) == 2)[0]
                if self.check(plays,self.wins):
                    print('AGENT WINS')
                    self.show_game_state(game_state)
                    break
                elif 0 not in game_state:
                    print('DRAW')
                    break
    
        
        
    def agent_play(self,game_state):

        what_state = list(self.states.values()).index(tuple(game_state))
        available_blocks = np.where(np.array(game_state) == 0)[0]
        available_q_vals =   np.array([self.Q[what_state][available_blocks],available_blocks])

        if len(np.unique(available_q_vals[0])) == 1:
            agent_action = np.random.choice(available_blocks)
        else:
            agent_action = available_q_vals[1,np.argmax(available_q_vals[0])]

        return self.actions[agent_action]
    

