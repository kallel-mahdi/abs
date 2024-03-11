
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from src.means.utils import *
from src.networks import *
import src.utils



def generate_batch(rng,size,n_steps,batch_size):
    
    return jax.random.choice(rng,a=size,shape=(n_steps,batch_size), replace=True)

class Agent:

    def __init__(self,

        state_dim,action_dim,
        actor_model,actor_params,param_reshaper,
        hidden_dims,policy_delay,polyak,
        use_layer_norm,activation_fn,dropout_rate,
        m,n,
        num_steps,critic_batch_size,
        normalize_obs,discount,
        seed):  
              
        ### Create critic architectures
        self.dummy_state = jnp.ones([1, state_dim])
        self.dummy_action = jnp.ones([1, action_dim]) 
        critic_base_cls = partial(MLP,
                            hidden_dims=hidden_dims,
                            activate_final=True,
                            dropout_rate=dropout_rate,
                            use_layer_norm=use_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        self.critic_def = Ensemble(critic_cls, num=n)
        
        ### Initialize critic weights
        critic_key = jax.random.PRNGKey(seed)
        self.critic_params = self.critic_def.init(critic_key,self.dummy_state,self.dummy_action)['params']
        self.critic_target_params = self.critic_def.init(critic_key, self.dummy_state,
                                        self.dummy_action)['params']
        self.critic_optimizer = optax.adam(learning_rate=3e-4) # TODO : switch to flax.trainstate
        
        ### Add normalization to critic inputs
        normalize_fn = src.utils.normalize if normalize_obs else   lambda x, y: x
        self.critic_def = make_normalized_qnetwork(self.critic_def,normalize_fn)
        
        ### Save local variables
        self.__dict__.update(locals())
        
        ### Useful functions
        self.a_states = self.actor_model.apply
        self.a_params = jax.vmap(self.a_states,in_axes=(param_reshaper.vmap_dict,None,None))
        self.q_states = self.critic_def.apply
        self.q_params = jax.vmap(self.q_states,in_axes=(None,None,0,None,None,None),out_axes=0) ## parallelize w.r.t actors
        
        self.fit_vmap = jax.jit(jax.vmap(self._fit_critic,in_axes=(0,0,
                                                     0,0,0,
                                                     None,None,None,
                                                     None)),
                                                    static_argnums=(8,))

        ############################
        self.generate_batch_vmap = jax.jit(jax.vmap(generate_batch,in_axes=(0,None,None,None)),static_argnums=(1,2,3))
        
        no_reset = lambda rng,params: params
        reset_critic = lambda rng,params : self.critic_def.init(rng,self.dummy_state,self.dummy_action)["params"]
        reset_opt = lambda rng,params : self.critic_optimizer.init(self.critic_params)
        cond_reset_critic = lambda  mask,rng,params :lax.cond(mask,reset_critic,no_reset,rng,params)
        cond_reset_opt = lambda mask,rng,params : lax.cond(mask,reset_opt,no_reset,rng,params)
        self.reset_critic_vmap = jax.jit(jax.vmap(cond_reset_critic,in_axes=(0,0,0)))
        self.reset_opt_vmap = jax.jit(jax.vmap(cond_reset_opt,in_axes=(0,0,0)))
        
        

    def _critic_loss(self,
            critic_params, 
            critic_target_params,
            actor_params,obs_params,
            transition,
            rng):
        
        param_reshaper = self.param_reshaper
        state, action,reward,next_state, not_done = transition
        
        ### Get next action
        actor_params = param_reshaper.reshape_single(actor_params.squeeze())
        next_action = self.a_states(actor_params,obs_params,next_state)
        
        ### Compute the target Q  value
        rng,key2,key3 = jax.random.split(rng,3)
        rng, key = jax.random.split(rng)
        all_indx = jnp.arange(0,self.n)
        indx = jax.random.choice(key, a=all_indx, shape=(self.m,), replace=False)
        
        ### Subsample from networks
        param_sample = jax.tree_util.tree_map(lambda param: param[indx],critic_target_params)
        next_qs = self.critic_def.apply({'params': param_sample},
                                   next_state,next_action,
                                   True,## Training
                                   rngs={'dropout':key2},
                                   obs_params=obs_params)##[n_params,batch_size]
        next_qs = next_qs[indx]
        
        ### Compute TD target
        next_q = jnp.min(next_qs, axis=0)
        target_q = reward.squeeze() + not_done.squeeze() * self.discount * next_q

        ### Compute current Q value
        qs = self.critic_def.apply({'params': critic_params},
                                       state,action,
                                       True,## Training
                                       rngs={'dropout': key3},
                                       obs_params=obs_params)
        
        ### Compute critic loss
        critic_loss = ((target_q-qs)**2).mean()
        
        return critic_loss


    
    @partial(jax.jit, static_argnames=["self"])
    def critic_step(self,
            carry, ## updatable_variables
            element,
            transitions,actor_params,obs_params,
            ):
        
        
        
        critic_opt_state,critic_params,critic_target_params,sum_loss,rng = carry
        i,batch_idx = element
        rng,_ = jax.random.split(rng)
        b_transitions = get_transition_batch(transitions,batch_idx)
        
        ### Update critic 
        vgrad_fn = jax.value_and_grad(self._critic_loss, argnums=0)
        loss, grad = vgrad_fn(
                critic_params, 
                critic_target_params,
                actor_params,obs_params,
                b_transitions,
                rng)
        
        sum_loss += loss
        updates, critic_opt_state = self.critic_optimizer.update(grad, critic_opt_state)
        critic_params = optax.apply_updates(critic_params, updates)
        
        ### Update critic_target periodically
        update = lambda x : soft_target_update(x,critic_params,self.polyak)
        no_update = lambda x: x
        critic_target_params = lax.cond(i%self.policy_delay==0,update,no_update,critic_target_params)
        
        ### Increment counter for transition batch
        i = i+1

        return (critic_opt_state,critic_params,critic_target_params,sum_loss,rng),_


    @partial(jax.jit, static_argnames=["self","n_steps"])
    def _fit_critic(self,rng,batch_idxs,
                    critic_params,critic_target_params,critic_opt_state,
                    transitions,actor_params,obs_params,
                    n_steps,):


            i=0
            sum_loss = 0
            critic_step = lambda carry,element : self.critic_step(carry,element,actor_params=actor_params,obs_params=obs_params,transitions=transitions)
            rslt,_ = jax.lax.scan(
            critic_step,
            ( critic_opt_state,critic_params,critic_target_params,sum_loss,rng),
            (jnp.arange(n_steps),batch_idxs),
            )
            
            critic_opt_state,critic_params,critic_target_params,sum_loss,rng = rslt
            
            return critic_opt_state,critic_params,critic_target_params,sum_loss


    def fit_critic(self,rng,
                   agent_state,
                   actor_params,obs_params,
                   transitions,
                   n_critics,R2_history,
                   new_actor,reset_critic):
        
        
                    
            
              

                    
                    x = agent_state.x
                    
                    ### Use low precision for training NN as this is the bottleneck
                    num_devices = jax.local_device_count()
                    rngs = jax.random.split(rng,n_critics)
                    num_steps = self.num_steps
                    
                    if new_actor :
                        num_steps = num_steps * 2
                    
                    
                    
                    reset = lambda rng,params : self.critic_def.init(rng,self.dummy_state,self.dummy_action)["params"]
                
                    ### Initially reset all nns (maybe to be handled in init agent)
                    if x == 0:
                        
                        print(f'first time')
                        mask = jnp.ones((n_critics,))
                        num_steps = 1000
                        params_pholder = jnp.ones((n_critics,))
                        b_critic_params = jax.vmap(reset,in_axes=(0,0))(rngs,params_pholder)
                        b_critic_target_params = jax.vmap(reset,in_axes=(0,0))(rngs,params_pholder)
                        b_critic_opt_state = jax.vmap(self.critic_optimizer.init,in_axes=0)(b_critic_params)
                    
                    ### Reset the worst performing critic  
                    if x!=0 : 
                        if new_actor :
                            
                            ### Reset all optimizers and worst critic
                            critic_mask = jnp.zeros((n_critics,))
                            

                            if reset_critic : 
                                 ### If a critic performs very poorly reset 
                                if jnp.min(R2_history) < 0 :
                                    critic_mask.at[jnp.argmin(R2_history)].set(1)
                                    print(f'resetting  {np.argmin(R2_history)}')
                                                                        
                                opt_mask = jnp.ones(n_critics)
                                b_critic_params = self.reset_critic_vmap(critic_mask,rngs,agent_state.b_critic_params)
                                b_critic_target_params = self.reset_critic_vmap(critic_mask,rngs,agent_state.b_critic_target_params)
                                b_critic_opt_state = self.reset_opt_vmap(opt_mask,rngs,agent_state.b_critic_opt_state)
                                print(f'R2  {R2_history}')
                            
                                
                            else :
                                
                                b_critic_opt_state = agent_state.b_critic_opt_state
                                b_critic_params = agent_state.b_critic_params
                                b_critic_target_params = agent_state.b_critic_target_params
                                print(f'R2  {R2_history}')
                                
                        
                        else :
                            
                            b_critic_opt_state = agent_state.b_critic_opt_state
                            b_critic_params = agent_state.b_critic_params
                            b_critic_target_params = agent_state.b_critic_target_params
                            print(f'R2  {R2_history}')
                    x +=1


                    ################# GPUs are optimizer for fp32 compute ##########################           
                    b_critic_params = jax.tree_map(lambda x:x.astype("float32"),b_critic_params)
                    b_critic_target_params = jax.tree_map(lambda x:x.astype("float32"),b_critic_target_params)
                    b_critic_opt_state = jax.tree_map(lambda x:x.astype("float32") if isinstance(x,jnp.floating) else x,b_critic_opt_state)

                    transitions = tuple([i.astype("float32") for i in transitions])
                    actor_params = jax.tree_map(lambda x:x.astype("float32"),actor_params)
                    obs_params = jax.tree_map(lambda x:x.astype("float32"),obs_params)
                    #################################################################################
                       
                    rngs = jax.random.split(rng,n_critics)
                    b_batch_idxs = self.generate_batch_vmap(rngs,agent_state.buffer_max_size,num_steps,self.critic_batch_size)
                    b_critic_opt_state,b_critic_params,b_critic_target_params,b_loss= self.fit_vmap(rngs,b_batch_idxs,
                                                                                                                            b_critic_params,b_critic_target_params,b_critic_opt_state,
                                                                                                                            transitions,actor_params,obs_params,
                                                                                                                            num_steps)
                    
                    agent_state = agent_state.replace(
                    b_critic_opt_state=b_critic_opt_state,
                    b_critic_params=b_critic_params,
                    b_critic_target_params=b_critic_target_params,
                    x=x,
                    )    
                           
                           

                    return agent_state,b_critic_target_params
