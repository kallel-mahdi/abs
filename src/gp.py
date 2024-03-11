from functools import partial

import jax
import jax.numpy as jnp
import jaxopt

from src.means.mean import ConstantMean
from src.utils import *




def generate_hps(rng,
                  l_b,u_b,
                  params_init:Callable):
  
    """ Generate random hyperparameters for the GP
    This is used to initialize the LBFGS loop"""
  
    sample_uniform = lambda param,l_b,u_b : jax.random.uniform(rng,minval=l_b,maxval=u_b,shape=param.shape)
    params =jax.tree_util.tree_map(lambda param,l_b,u_b : sample_uniform(param,l_b,u_b),params_init,l_b,u_b)
  

    return params



def likelihood1(params,
                mean_state,kernel_state,
                X_prep,y,mean_vec,
                mean_call:Callable,kernel_forward_inputs_hps:Callable):
  
    """ This one computes the likelihood of the GP with a call to the mean function"""
  
    n = X_prep.shape[0]
    
    # Covariance matrix K[X,X] with noise measurements
    K = kernel_forward_inputs_hps(params,kernel_state,X_prep,X_prep)

    # Take out mean prediction from targets
    y = y - mean_call(params,mean_state,X_prep)
    K_inv_y = jax.scipy.linalg.solve(K,y,assume_a="pos")
    fitting = jnp.dot(y.T,K_inv_y)
    sign,penalty = jnp.linalg.slogdet(K)
    
    log_likelihood = -0.5 * jnp.sum(fitting + penalty + n * jnp.log(2.0 * jnp.pi))

  
    return -log_likelihood
  
  
  

def likelihood2(params,
                mean_state,kernel_state,
                X_prep,y,mean_vec,
                mean_call:Callable,kernel_forward_inputs_hps:Callable):
  
    """ This one computes the likelihood of the GP using a cached call to the mean function
    Indeed, calling during the optimization of the hyperparameters of the GP, the mean function of ABS
    remains unchanged (it's parameters are those of the critic). This will avoid calling the NN and 
    speeds up optimizing the GP hyperparameters."""
  
    n = X_prep.shape[0]
    # Covariance matrix K[X,X] with noise measurements
    K = kernel_forward_inputs_hps(params,kernel_state,X_prep,X_prep)
    y= y - mean_vec

    K_inv_y = jax.scipy.linalg.solve(K,y,assume_a="pos")
    fitting = jnp.dot(y.T,K_inv_y)
    sign,penalty = jnp.linalg.slogdet(K)

    log_likelihood = -0.5 * jnp.sum(fitting + penalty + n * jnp.log(2.0 * jnp.pi))

    return -log_likelihood
  
  
  



def _fit(rng,
          params,mean_state,kernel_state,
          X,y,
          bounds,
          hps_optim:Callable,generate_hps:Callable,mean_call : Callable,kernel_prepare_inputs: Callable):

      """ fit hyperparameters of GP
      Using any other optimizer than L-BFGS-B will cause years of tears and existential crisis"""
    
    

      # initialize loop states
      mean_vec = mean_call(params,mean_state,X) ### This will be only used in the case of AdvMean
      params = generate_hps(rng,bounds[0],bounds[1])
      X_prep,_ = kernel_prepare_inputs(kernel_state,X,X) 
      bounds
      rslt = hps_optim.run(init_params=params,
                                bounds = bounds,
                                mean_state=mean_state,kernel_state=kernel_state,
                                X_prep=X_prep,y=y,mean_vec=mean_vec)
      params = rslt.params
      likelihood = rslt.state.value
   
      return params,likelihood
  

  
class GP(object):


  def __init__(self,mean,kernel):
    

    """
    Base function that deals with the Gaussian Processes.
    """
    # Number of points in the prior distribution

    self.__dict__ = locals()
    self.X = None
    self.y = None
    self.states = None
    self.masks = None
    self.x = None ### Local policy
    self.noise = None ### Local noise
    self.params,self.l_b,self.u_b = {},{},{}
    
    self.mean_state = mean.state
    self.kernel_state = kernel.state
    
    
    """"For GP Class we combine the parameters of mean and kernel
    to be able to optimize w.r.t to joint parameters"""
    self.params = self.kernel.init_params.copy()
    self.l_b.update(kernel.l_b)
    self.u_b.update(kernel.u_b)

    #################
    
    if isinstance(mean,ConstantMean):
      
      ### Add mean parameters
      self.params.update(mean.init_params)
      self.params_init = self.params.copy()
      self.l_b.update(mean.l_b)
      self.u_b.update(mean.u_b)
      self.is_constant_mean = True
      self.generate_hps = partial(generate_hps,params_init=self.params_init)
      self.likelihood = partial(likelihood1,mean_call=self.mean.__call__,
                                          kernel_forward_inputs_hps=self.kernel._forward_inputs_hps)    

    else : 
      self.params_init = self.params.copy()
      self.generate_hps = partial(generate_hps,params_init=self.params_init)
      self.likelihood = partial(likelihood2,mean_call=self.mean.__call__,
                                          kernel_forward_inputs_hps=self.kernel._forward_inputs_hps)
      
      self.is_constant_mean = False
    
    self.hps_optim = jaxopt.LBFGSB(fun=self.likelihood)
    
    self._fit = partial(_fit,
                        hps_optim=self.hps_optim,
                        mean_call=self.mean.__call__,kernel_prepare_inputs=self.kernel._prepare_inputs,generate_hps=self.generate_hps)
    
    self.fit_parallel_vmap = jax.jit( jax.vmap(self._fit,in_axes=(0,
                                                         None,None,None,
                                                         None,None,
                                                         None)))
    
    f = lambda b_critic_params,argmax : jax.tree_map(lambda param: param[argmax],b_critic_params)
    self.fetch_best = jax.jit(f)
    
    self.l_b = jax.tree_map(lambda x,y: x*jnp.ones_like(y),self.l_b,self.params_init)
    self.u_b = jax.tree_map(lambda x,y: x*jnp.ones_like(y),self.u_b,self.params_init)
    
    self.l_b_original = self.l_b.copy()
    
      
      


  def fit(self,rng):
    
    ### You cant fit with single point
    if self.X is None :
      pass
    elif self.X.shape[0]==1:
      pass
    
    else :
      
     
      y_pred = self.mean.__call__(self.params,self.mean_state,self.X)
      self.signal_variance = (self.y-y_pred).std()
      self.signal_noise = self.noise[self.noise!=0].mean()
      self.snr = self.y.std()/self.signal_noise
      ### Avoid nan issues (0 noise)
      if jnp.isnan(self.signal_noise):
      
        self.signal_noise = 1e-4
      
      self.l_b["noise"]= (1/3) * self.signal_noise
      self.u_b["noise"]= 3 * self.signal_noise
      self.l_b["outputscale"] = (1/3) * self.signal_variance
      self.u_b["outputscale"] = 3 * self.signal_variance

      rngs = jax.random.split(rng,32)
      
      params_b,likelihoods_b = self.fit_parallel_vmap(rngs, self.params,
                                                  self.mean_state,self.kernel_state,
                                                  self.X,self.y,
                                                  (self.l_b,self.u_b))
      argmax = jnp.nanargmax(-likelihoods_b)
      params_max = self.fetch_best(params_b,argmax)
      likelihood_max = likelihoods_b[argmax]


      # set new parameters
      self.params = params_max.copy()
    
      return params_max,likelihood_max


  def append_data(self,X_new,y_new,noise_new,
                  states_new,masks_new,n_max=None):
    
    
    

    if self.X is None :
      
      self.X = X_new
      self.y = y_new.reshape(-1,1)
      self.noise = noise_new.reshape(-1,1)
      self.states = states_new
      self.masks = masks_new
    
    else : 
      
      if n_max == None:
        
      
        self.X = jnp.vstack([self.X,X_new])
        self.y = jnp.vstack([self.y,y_new])
        self.noise = jnp.vstack([self.noise,noise_new])
        self.states = jnp.vstack([self.states,states_new])
        self.masks = jnp.vstack([self.masks,masks_new])
      
      else : 
        
        self.X = jnp.vstack([self.X,X_new])[-n_max:]
        self.y = jnp.vstack([self.y,y_new])[-n_max:]
        self.noise = jnp.vstack([self.noise,noise_new])[-n_max:]
        self.states = jnp.vstack([self.states,states_new])[-n_max:]
        self.masks = jnp.vstack([self.masks,masks_new])[-n_max:]  
        
  
