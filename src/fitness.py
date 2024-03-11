import gymnasium 
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
import src.utils
import logging
#import gym_lqr
import pickle

from src.utils import CodeTimer

class SkipDone(gymnasium.Wrapper):

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_state = None
        self._is_done = False
        return observation, info

    def step(self, action):


        if not self._is_done:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._is_done = terminated or truncated
            info["_terminated"] = terminated
            info["_truncated"] = truncated


            terminated = truncated = False  # To avoid being autoreset.
            self._last_state = (observation, reward, terminated, truncated, info)
            

        return self._last_state


class PostProcessSkipDone(gymnasium.experimental.vector.VectorWrapper):
    
    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = self.env.step(actions)
        terminateds = infos["_terminated"]
        truncateds = infos["_truncated"]
        return observations, rewards, terminateds, truncateds, infos



class BraxFitness(object):
    def __init__(
        self,
        env_name,
       reward_shift,
       reward_scale,
       discount,
       normalize_obs,
       seed,
       **kwargs,
    ):
        


        def create_envs(env_name,discount,reward_shift,reward_scale,num_envs,seed):
        
            NormReward = partial(gymnasium.wrappers.TransformReward,f = lambda r : (r-reward_shift)/reward_scale)
            envs = gymnasium.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={"shared_memory": False},
            wrappers=[SkipDone,NormReward],
            )
            
            
            envs = PostProcessSkipDone(envs)
            envs.discount = discount
            envs._max_episode_steps = 1000
            envs.reset(seed=int(seed))

            

            return envs

        
        self.env = gymnasium.make(env_name)
        self.seed = seed
        
     
        self.action_shape = self.env.action_space.shape[0]

        
        self.input_shape = self.env.observation_space.shape
        self.train_envs = partial(create_envs,env_name=env_name,discount=discount,
                                   reward_shift=reward_shift,reward_scale=reward_scale,seed=self.seed)
        
        
        self.eval_envs = partial(create_envs,env_name=env_name,discount=1.,
                                   reward_shift=0.,reward_scale=1.,seed=self.seed)
        
        
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.discount = discount
        self.normalize_obs = normalize_obs
        self.seed = seed
        self.env_dict = {}
        
        
        self.normalizer_params = src.utils.init_state(src.utils.Array(self.input_shape, jnp.float32))
        

    def set_apply_fn(self, map_dict, network_apply, carry_init=None):
            
            """Set the network forward function."""
            
            ######################################################
            self.network = network_apply
            self.vmapped_policy = jax.jit(jax.vmap(network_apply, in_axes=(map_dict,None,0)),backend="cpu") ## take many params
            self.env_reset = self.env.reset
           
            

    def run_episodes(self,
                     param_reshaper,policy_params,normalizer_params,eval=False):


        
        with jax.default_device(jax.devices("cpu")[0]):
                

            n_envs = policy_params.shape[0]
            policy_params =param_reshaper.reshape(policy_params) 
            
            
            if (eval,n_envs) in self.env_dict.keys() :
                envs = self.env_dict[(eval,n_envs)]
            
            else :
                
                if eval :
                        envs = self.eval_envs(num_envs=n_envs)
                    
                else : 
                        envs = self.train_envs(num_envs=n_envs)
                
                self.env_dict[(eval,n_envs)]=envs
                    
                
                
                
            ep_observation,ep_action,ep_reward,ep_next_observation,ep_terminated,ep_take = [],[],[],[],[],[]
            observation, info = envs.reset()
            take = np.ones((n_envs,))
            

            for i in range(1000):
                
                action = self.vmapped_policy(policy_params,normalizer_params,observation)
                next_observation, reward, terminated, truncated, info = envs.step(action)

                ep_observation.append(observation)
                ep_next_observation.append(next_observation)
                ep_action.append(action)
                ep_reward.append(reward)
                ep_terminated.append(terminated*1)
                ep_take.append(take)
                observation = next_observation
                take = take * (1-terminated)

            #envs.close()
                

            discount = np.array([envs.discount**i for i in range(1000)])
            ep_observation = np.transpose(np.stack(ep_observation),(1,0,2))
            ep_next_observation = np.transpose(np.stack(ep_next_observation),(1,0,2))
            ep_action = np.transpose(np.stack(ep_action),(1,0,2))
            ep_reward = np.vstack(ep_reward).T
            ep_terminated = np.vstack(ep_terminated).T
            ep_take = np.vstack(ep_take).T
            ################
            ep_reward = ep_reward * ep_take
            ################
            ep_mask = (1- ep_terminated)*discount
            cummulative_reward = (ep_reward * ep_mask).sum(axis=1)
            
            ep_s = ep_observation.reshape(-1,ep_observation.shape[-1])
            ep_next_s = ep_next_observation.reshape(-1,ep_next_observation.shape[-1])
            ep_action = ep_action.reshape(-1,ep_action.shape[-1])
            ep_reward = ep_reward.reshape(-1,1)
            ep_not_done = (ep_mask!=0).reshape(-1,1)
            ep_active = (ep_take).reshape(-1,1)
            
            #s,a,r,next_s,not_done,buffer_take
            transitions = (ep_s,ep_action,ep_reward,ep_next_s,ep_not_done,ep_active)
            
            
            

            
            return cummulative_reward.reshape(-1,1),ep_observation,ep_mask,transitions
        






def fitness_acq(evaluator,param_reshaper,policy_params,normalizer_params,eval=False,num_rollouts=1):

    n_policies = policy_params.shape[0]
    policy_params = jnp.repeat(policy_params,axis=0,repeats=num_rollouts)
    reward_n,obs_n,mask_n,transitions_n= evaluator.run_episodes(param_reshaper,policy_params,normalizer_params,eval)
    
    reward = reward_n.reshape(n_policies,num_rollouts,1).mean(axis=1)
    signal_noise = reward_n.reshape(n_policies,num_rollouts,1).std(axis=1)
    episode_length = obs_n.shape[1]
    obs = obs_n.reshape(n_policies,num_rollouts*episode_length,-1)
    mask = mask_n.reshape(n_policies,num_rollouts*episode_length)

    transitions = transitions_n
    
    
    
    return reward,obs,mask,transitions,signal_noise




def fitness_center(evaluator,
                param_reshaper,param,normalizer_params,
                num_rollouts,eval=False):
    
    param = param.squeeze().reshape((1,-1))
    param_batch = jnp.repeat(param,num_rollouts,axis=0)
    cum_reward, obs, obs_weights, transitions = evaluator.run_episodes(param_reshaper,param_batch,normalizer_params,eval)
    noise = cum_reward.std()
    cum_reward = cum_reward.mean(axis=0)

    return cum_reward,noise,obs, obs_weights, transitions 






def update_norm(normalizer_params,obs,obs_weights):
    
    return src.utils.update(normalizer_params,obs, 
        weights=obs_weights)


