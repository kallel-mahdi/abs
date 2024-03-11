######## Buffer ####
import flax.linen as nn
import jax
import jax.lax as lax
import numpy as np
import optax
import timeit
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from jax import numpy as jnp
from jax import random as jrandom



def get_transition_batch(transitions,indx):
    
    
    ### for the first itieration buffer size may be smaller than batch size
    state, action, reward,next_state, not_done, = transitions
    b_state = state[indx]
    b_action = action[indx]
    b_reward = reward[indx]
    b_next_state = next_state[indx]
    b_not_done = not_done[indx]
    
    return (b_state,b_action,b_reward,b_next_state,b_not_done,)



class ReplayBuffer(object):
    
    def __init__(self, state_dim, action_dim,max_size=int(1e5)):
        
        
        self.max_size = max_size
        self.keys = PRNGKeys(seed=0)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.not_done = np.zeros((max_size, 1))
        self.len_transitions = [100,100]


    def add_batch(self,transitions):
        
        count = 0
        for s,a,r,next_s,not_done,take in zip(*transitions):
            
            if take :
                self.add(s,a,r,next_s,not_done)
                count+=1
                
                
        self.len_transitions.append(count)
        
        
    def add(self, state, action, reward, next_state,not_done,):
        
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.not_done[self.ptr] = not_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    
    def sample(self,batch_size):
        
        
        key = self.keys.get_key()
        indx = np.array(jax.random.randint(key,(batch_size,),0,self.size))
        
        return (
            jax.device_put(self.state[indx]),
            jax.device_put(self.action[indx]),
            jax.device_put(self.reward[indx]),
            jax.device_put(self.next_state[indx]),
            jax.device_put(self.not_done[indx])
        )
        
    def get_all_transitions(self):
        
        """ Return fixed arrays of size max_size containing all the transitions available"""
        
     
       
        if self.size < self.max_size : 
            key = self.keys.get_key()
            indx = np.array(jax.random.choice(key,shape=(self.max_size,),a=self.size,replace=True))
            
            return (
            jax.device_put(self.state[indx]),
            jax.device_put(self.action[indx]),
            jax.device_put(self.reward[indx]),
            jax.device_put(self.next_state[indx]),                
            jax.device_put(self.not_done[indx]),
        )
              
        else : 
            
            return (
            jax.device_put(self.state),
            jax.device_put(self.action),
            jax.device_put(self.reward),
            jax.device_put(self.next_state),                
            jax.device_put(self.not_done)
        )
        








class PRNGKeys:

    def __init__(self, seed=0):
        self._key = jrandom.PRNGKey(seed)

    def get_key(self):
        self._key, subkey = jrandom.split(self._key)
        return subkey



def soft_target_update(
                        target_critic_params: FrozenDict,
                        critic_params: FrozenDict,                   
                        tau: float) -> FrozenDict:
    
    new_target_params = jax.tree_map(lambda  tp,p:  tp * (1 - tau) + p * tau,
                                     target_critic_params,critic_params )

    return new_target_params


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start)
        print('Code block' + self.name + ' took: ' + str(self.took) + ' s')

