import jax
import optax

from src.kernels import *
from src.utils import CodeTimer
from src.means.mean import ConstantMean



def gradient_step_mpd(self,x,X,y):

    
    ascent_prob = 1.
    
    num_iters = 0
    
    
    while ascent_prob > 0.65 and num_iters<1000:

      mean_d, variance_d = self.gp.posterior_derivative(x)
      ascent_direction = jax.scipy.linalg.solve(variance_d,mean_d.squeeze(),assume_a ="pos")
      ascent_prob = jax.scipy.stats.norm.cdf(-jnp.dot(mean_d,-ascent_direction)/jnp.sqrt(ascent_direction@variance_d@ascent_direction))
      ### In their code, MPD normalize the ascent direction
      ### Line 1002 : https://github.com/kayween/local-bo-mpd/blob/main/src/optimizers.py
      ascent_direction = self.learning_rate * (ascent_direction/jnp.linalg.norm(ascent_direction))
      x += ascent_direction
      num_iters +=1
    
    
    print(f'num_iters {num_iters}')
    
    
    return x,ascent_prob



def gradient_step_abs(self,x,X,y):

    mean_d, variance_d = self.gp.posterior_derivative(x)
    ascent_direction = jax.scipy.linalg.solve(variance_d,mean_d.squeeze(),assume_a ="pos")
    ascent_prob = jax.scipy.stats.norm.cdf(-jnp.dot(mean_d,-ascent_direction)/jnp.sqrt(ascent_direction@variance_d@ascent_direction))
    ascent_direction = self.learning_rate*ascent_direction
    
    norm = jnp.mean(jnp.abs(ascent_direction))
    if norm >self.max_grad:
      ascent_direction = (self.max_grad / (jnp.mean(jnp.abs(ascent_direction)))) * (ascent_direction)

    x_new = x + ascent_direction
    
    return x_new,ascent_prob
  


class MPD:

  def __init__(self,
               gp,
               n_info,n_max,n_parallel,
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

  
  

  def gradient_step(self,x,X,y):
    
    
        
    if isinstance(self.gp.mean,ConstantMean):
      
      #self.gradient_step = partial(gradient_step_mpd,self=self)
      return gradient_step_mpd(self,x,X,y)
    
    else : 
      
      return gradient_step_abs(self,x,X,y)


    


  def ask_neighbours(self,rng,m=None):
    
    """`ask` for new parameter candidates to evaluate next."""
    
    m = self.n_parallel
    
    
    with CodeTimer("Gp ask neighbours"):
      
      
      X_acq,X_acq_info,info_gain = self.gp.get_query_pts(rng,self.x,m)

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
  
    ### Hotfix for occasional nan grads
    if self.x is None or not jnp.array_equal(self.x,x_t):
      
      self.gp.append_data(x_t,y_t,noise_t,
                          ### Fix to stack states_t and states_acq
                          s_t_r[:,:self.num_rollouts*1000,:],mask_t_r[:,:self.num_rollouts*1000],self.n_max)
      self.x = x_t
      self.y_t = y_t
      self.gp.noise_t= noise_t
    
      
    
    if not self.gp.is_constant_mean :
      

        old_params = self.gp.mean_state
        tmp_params = self.gp.mean.set_train_data(old_params,x_t,y_t,s_t,mask_t,transitions_t)
        
        with CodeTimer("Fit critic on new policy"):
          self.gp.mean_state = self.gp.mean.fit_critic(rng,tmp_params,
                                                        self.gp.X,self.gp.y,self.gp.states,self.gp.masks,new_actor=True)



    with CodeTimer("Gp fit new policy"):
      
      if not self.first_step :
        self.gp.fit(rng)
      
      else :
        self.first_step = False
      
    
    
      

  def tell_neighbours(self,rng,X_acq,y_acq,noise_acq,
                      states_acq,masks_acq,transitions_acq):

  
    self.gp.append_data(X_acq,y_acq,noise_acq,
                        states_acq,masks_acq,self.n_max)
    


  
    if not self.gp.is_constant_mean:
      
      
        
                  
      states_acq = states_acq.reshape((-1,*states_acq.shape[2:]))
      masks_acq = masks_acq.reshape((-1,*masks_acq.shape[2:]))
      self.gp.mean.append_train_data(X_acq,y_acq,states_acq,masks_acq,transitions_acq)        
      
      old_params = self.gp.mean_state

      with CodeTimer("Fit critic on new policy"):
        
        self.gp.mean_state = self.gp.mean.fit_critic(rng,old_params,
                                                      self.gp.X,self.gp.y,self.gp.states,self.gp.masks,new_actor=False)
      
    with CodeTimer("Gp fit neighbours"):
      
      self.gp.fit(rng)
      
        
          
      


      
    
  


    



