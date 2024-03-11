from functools import partial
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from src.means.utils import *
from jax.config import config as cfg


@struct.dataclass
class AgentState:
    
    buffer_max_size : int
    b_critic_params : chex.Array = None
    b_critic_target_params : chex.Array = None
    b_critic_opt_state : chex.Array = None
    x : int = 0


@struct.dataclass
class AdvantageMeanState:
    
    num_rollouts_center : chex.Array 
    
    obs_params : chex.Array
    agent_state :  AgentState 
    
    critic_weights : Optional[chex.Array] = None
    local_actor : Optional[chex.Array]=None ## local_actor
    local_transitions : Optional[chex.Array]=None
    local_states : Optional[chex.Array]=None
    local_masks  : Optional[chex.Array]=None    
    local_return : Optional[chex.Array]=None



@partial(jax.jit,static_argnames=["param_reshaper","a_states","q_states"])
def get_local_q(    local_actor_params,
                    critic_params,
                    obs_params,
                    local_states,
                    param_reshaper,
                    a_states:callable,
                    q_states:callable):
    
        
        
        local_actor_params  = param_reshaper.reshape_single(local_actor_params)
        local_actions = a_states(local_actor_params,obs_params, local_states)
        local_q = q_states({'params': critic_params},
                                       local_states,local_actions,
                                       training=False,
                                       rngs=None,
                                       obs_params=obs_params
                                        ) ## [n_crtics,n_states] ## n_critics aka 2
        
        
        local_q = jnp.mean(local_q, axis=0) ## [n_states]
        
        local_q = local_q.reshape(1,-1) ## [1,n_states] because acq_q [N_actors,n_states]
        
        
        return local_q


@partial(jax.jit,static_argnames=["param_reshaper","a_params","q_params"])
def get_acq_q(
            acq_actors_params,
            critic_params,
            obs_params,
            local_states,
            
            param_reshaper,
            a_params:callable,
            q_params:callable,
            ):
    
    acq_actors_params = param_reshaper.reshape(acq_actors_params)
    ### actions_acq 
    a_acq = a_params(acq_actors_params,obs_params,local_states) #[n_states,action_dim,n_params]
    
    ### q_acq [n_params,n_states,1]
    q_acq =q_params ({'params': critic_params},
                                    local_states,a_acq,
                                    False,#training
                                    None,#rngs
                                    obs_params
                                    ) ## [n_params,n_critics,n_states] ##n_critics as in 2 :))
                        

    q_acq = jnp.mean(q_acq,axis=1)## [n_params,n_states]
    
    return q_acq



@partial(jax.jit,static_argnames=["get_local_q","get_acq_q"])
def call_one_critic(mean_params,mean_state,X,critic_params,
            get_local_q : callable,
            get_acq_q : callable,
            ):
    

    local_q = get_local_q(mean_state.local_actor,critic_params,
                          mean_state.obs_params,mean_state.local_states,)
    
    acq_q = get_acq_q(X,critic_params,
                      mean_state.obs_params,mean_state.local_states,)
    
    ### Take only non pit state
    advantage = (acq_q-local_q)*(mean_state.local_masks) ### advantage [n_params,n_states]        
    ### Average on rollouts
    advantage = advantage.sum(axis=1)/mean_state.num_rollouts_center ##[n_params]   
    
    return (mean_state.local_return + advantage).squeeze()


@partial(jax.jit,static_argnames=["call_many_critics"])
def __call__(mean_params,mean_state,X,
             call_many_critics:callable):
    
    
    b_critic_params = mean_state.agent_state.b_critic_target_params
    
    critic_preds = call_many_critics(mean_params,mean_state,X,b_critic_params,)
    
    weighted_pred = jnp.dot(mean_state.critic_weights.squeeze(),critic_preds.squeeze())
    
    
    return weighted_pred.squeeze()
    


def call_validation(mean_state,critic_params,
                            policy,states,masks,
                            call_one_critic:callable,num_rollouts:int):
            
             
            mean_state = mean_state.replace(
                 local_states = states,
                 local_masks = masks,
                 num_rollouts_center=num_rollouts)

            policy = policy.reshape((1,-1))
            y_pred = call_one_critic(None,mean_state,policy,critic_params).squeeze()

            return y_pred



def evaluate_one_critic(mean_state,critic_params,
                            X_eval,y_eval,states_eval,masks_eval,
                            call_validation):


            y_pred = jax.vmap(call_validation,in_axes=(None,None,0,0,0))(
                            mean_state,critic_params,
                            X_eval,states_eval,masks_eval)
            
            y_pred = y_pred.squeeze()
            y_eval = y_eval.squeeze()
            a2 = jnp.clip(((y_eval-y_pred)**2),a_min=1e-4).sum()
            b2=((y_eval-y_eval.mean())**2).sum()+1e-6
            R2 = 1-(a2/b2)
            
            return y_pred,R2
    


class AdvantageMean:
    
    def __init__(self,agent,obs_params,
                 num_rollouts_center,num_rollouts,
                 n_critics,buffer_max_size,
                 state_dim,action_dim,
                 aggregation,reset_critic) -> None:
        
        
        ### AdvMean does not require updates w.r.t likelihood
        self.params = {}

        agent_state = AgentState(buffer_max_size)
        self.state = AdvantageMeanState(
                    num_rollouts_center=num_rollouts_center,
                    obs_params=obs_params,
                    agent_state=agent_state) ## initialised later
        
        
        self.agent = agent
        self.reset_critic = reset_critic 
        self.n_critics = n_critics
        self.aggregation = aggregation
        self.R2_history = jnp.zeros((self.n_critics,))
     
        self.replay_buffer =   ReplayBuffer(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    max_size=buffer_max_size,
                    )
        
        
        self.get_local_q = partial(get_local_q,
                                   param_reshaper=self.agent.param_reshaper,a_states=self.agent.a_states,q_states=self.agent.q_states)
        
        self.get_acq_q = partial(get_acq_q,
                                   param_reshaper=self.agent.param_reshaper,a_params=self.agent.a_params,q_params=self.agent.q_params)
        
        self.call_one_critic = partial(call_one_critic,get_local_q=self.get_local_q,get_acq_q=self.get_acq_q)
        
        call_many_critics = jax.jit(jax.vmap(self.call_one_critic,in_axes=(None,None,None,0)))
        
        self.__call__ = partial(__call__,call_many_critics=call_many_critics)
        

        self.call_validation = partial(call_validation,call_one_critic=self.call_one_critic,num_rollouts=num_rollouts)
        
        self.evaluate_one_critic = partial(evaluate_one_critic,call_validation=self.call_validation)
        
        self.evaluate_one_critic_jit = jax.jit(self.evaluate_one_critic)        

        self.evaluate_vmap = jax.jit(jax.vmap(self.evaluate_one_critic,in_axes=(None,0,
                                                                                None,None,None,None,
                                                                                )))

    def set_train_data(self,state,
                       local_actor,local_return,local_states,local_masks,local_transitions,):
        
    
        ### Add transitions to buffer
        self.replay_buffer.add_batch(local_transitions)
        
        ### Update mean params
        new_state = state.replace(
                            local_return=local_return,
                            local_states=local_states,
                            local_masks=local_masks.squeeze(),
                            local_actor=local_actor.squeeze(),
                            
                            )
       
        return new_state
    
    
    def append_train_data(self,acq_params,acq_return,acq_states,acq_masks,acq_transitions):
        
        cpus = jax.devices("cpu") 
        acq_transitions = [jax.device_put(i,cpus[0]) for i in acq_transitions]
        
        self.replay_buffer.add_batch(acq_transitions)
        

    def fit_critic(self,rng,mean_state:AdvantageMeanState,
                   X,y,states,masks,new_actor)-> AdvantageMeanState:
        


        cfg.update("jax_enable_x64", False)
        cfg.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)        
        X = X.astype("float32")
        y = y.astype("float32")
        states = states.astype("float32")
        masks = masks.astype("float32")        
            
        
        ### Fit critic because we collected new transitions from neighbours
        agent_state,b_critic_params = self.agent.fit_critic(rng,
                                                mean_state.agent_state,
                                                mean_state.local_actor,mean_state.obs_params,
                                                self.replay_buffer.get_all_transitions(),
                                                self.n_critics,self.R2_history,
                                                new_actor,self.reset_critic)
        
        mean_state = mean_state.replace(agent_state=agent_state)
        preds,b_R2 = self.evaluate_vmap(mean_state,b_critic_params,X,y,states,masks)
        b_y_pred,b_R2 = self.evaluate_vmap(mean_state,b_critic_params,X,y,states,masks)
        

        if self.aggregation == "softmax":
            critic_weights = jax.nn.softmax(b_R2)
                    
        elif self.aggregation == "mean":
            
            critic_weights = jnp.ones_like(b_R2)
            critic_weights = critic_weights / critic_weights.sum()
        
        elif self.aggregation == "max":
            
            critic_weights = jnp.zeros_like(b_R2)
            critic_weights = critic_weights.at[jnp.argmax(b_R2)].set(1)
            
        
        #####################################
        self.R2_history = b_R2 
        new_mean_state = mean_state.replace(critic_weights = critic_weights,)

        cfg.update("jax_enable_x64", True)
        cfg.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)        
    
        
        return new_mean_state

         
    


def cst(C,x):
  return jax.vmap((lambda x : C))(x)


class ConstantMean():

  def __init__(self,constant_prior):

  

    self.init_params = params = {"constant": jnp.array(0.)}
    self.state = None
    
    self.l_b = {
        "constant":constant_prior.a,
    }

    self.u_b = {
        "constant":constant_prior.b,
    }
    
  
  def __call__(self,mean_params,mean_state,X):

    return cst(mean_params["constant"],X).squeeze()

