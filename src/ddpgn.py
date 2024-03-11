import jax
import optax

from src.kernels import *
from src.utils import CodeTimer



"""Unused code (for now)

This is a baseline where we disturb the parameters of the linear actor using a Gaussian noise 
and then follow the deterministic policy gradient.
Basically,this is MPD without smart acquisition and without posterior gradient correction"""

class DDPGN:

  def __init__(self,
               gp,
               n_info,n_max,
               x_sample,learning_rate,max_grad,
               num_rollouts,num_rollouts_center,num_learning_steps,
               *args,**kwargs):
    


    # Initialize optimizer
    total_steps = 200
    d = x_sample.shape[-1]

    n_samples = 0
    n_dims = x_sample.shape[-1]
    self.__dict__.update(locals())
    self.learning_rate = learning_rate
    self.max_grad = max_grad
    self.call = 0 
    self.x = None
    self.first_step = True
    self.sigma = self.gp.sigma
    self.n_parallel = n_info  ### All acquisition points are sampled in paralle

  def gradient_step(self,x,X,y):
    
    
    params = self.gp.params
    mean_state = self.gp.mean_state
    ascent_direction = jax.grad(self.gp.mean.__call__,argnums=2)(params,mean_state,x).squeeze()
    ascent_direction = self.learning_rate*ascent_direction
    
    # norm = jnp.mean(jnp.abs(ascent_direction))
    # if norm >self.max_grad:
    #   print(f'normaliziiing norm {norm}')
    #   ascent_direction = (self.max_grad / (jnp.mean(jnp.abs(ascent_direction)))) * (ascent_direction)
    
    x += ascent_direction

    return x,None


  def ask_neighbours(self,rng,m=None):
    
    """`ask` for new parameter candidates to evaluate next."""
    
    x = self.x 

    pos_noise = jax.random.normal(rng, (self.n_info//2,self.n_dims))
    all_noise = jnp.vstack([pos_noise,-pos_noise])
    
    X_acq = x +  self.sigma * all_noise
    
    #### Placeholders
    X_acq_info = 0
    info_gain = 0  
    
    self.noise = pos_noise
    

    return X_acq,X_acq_info,info_gain
  

  def ask_local(self):

    x_old = self.x
    
    x_new,ascent_prob = self.gradient_step(x_old,self.gp.X,self.gp.y)
  
    return x_new,ascent_prob

  

  def tell_local(self,rng,x_t,y_t,noise_t,
                 s_t,mask_t,transitions_t,obs_params=None):
    

    s_t = s_t.reshape((-1,*s_t.shape[2:]))
    mask_t = mask_t.reshape((-1,*mask_t.shape[2:]))
    s_t_r= jnp.expand_dims(s_t,axis=0)
    mask_t_r = jnp.expand_dims(mask_t,axis=0)      
    self.gp.append_data(x_t,y_t,noise_t,
                        ### Fix to stack states_t and states_acq
                        s_t_r[:,:self.num_rollouts*1000,:],mask_t_r[:,:self.num_rollouts*1000],self.n_max)
    self.x = x_t
    self.y_t = y_t
    ### Placeholders just to keep things running ###
    self.gp.noise_t= 0
    self.gp.signal_noise = 0
    self.gp.signal_variance = 0
    ################################################


    old_params = self.gp.mean_state
    tmp_params = self.gp.mean.set_train_data(old_params,x_t,y_t,s_t,mask_t,transitions_t)
    
    with CodeTimer("Fit critic on new policy"):
      self.gp.mean_state = self.gp.mean.fit_critic(rng,tmp_params,
                                                    self.gp.X,self.gp.y,self.gp.states,self.gp.masks,new_actor=True)


  def tell_neighbours(self,rng,X_acq,y_acq,noise_acq,
                      states_acq,masks_acq,transitions_acq):

  
    self.gp.append_data(X_acq,y_acq,noise_acq,
                        states_acq,masks_acq,self.n_max)
     
    states_acq = states_acq.reshape((-1,*states_acq.shape[2:]))
    masks_acq = masks_acq.reshape((-1,*masks_acq.shape[2:]))
    self.gp.mean.append_train_data(X_acq,y_acq,states_acq,masks_acq,transitions_acq)        
    
    ### Fit mean function as much as we would've in MPD + ADV
    for i in range(self.n_parallel):
      
      old_params = self.gp.mean_state
      self.gp.mean_state = self.gp.mean.fit_critic(rng,old_params,
                                                    self.gp.X,self.gp.y,self.gp.states,self.gp.masks,new_actor=False)
    
        
          
      


      
    
  


    




