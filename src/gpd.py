from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
from jax.scipy.linalg import cholesky,solve
from src.utils import *
from src.gp import GP

@partial(jax.jit)
def fetch_best(Z_b,Z_info_b,info_gain_b):
  
  argmax = jnp.nanargmax(-Z_info_b)
  Z_max = Z_b[argmax]
  Z_info_max = -Z_info_b[argmax]
  info_gain_max = info_gain_b[argmax]
  
  return Z_max,Z_info_max,info_gain_max
  

def info(
         Z,
         all_params,
         mean_state,kernel_state,   
         x_t,X,y,      
         K_XX,L_XX,K_XX_inv,K_xX_dx,mean_d,
         mean:Callable,kernel:Callable
         
         ):
      

      ##### mpd FACTORS #####
      d = X.shape[-1]
      Z = Z.reshape(Z.shape[0]//d,d)
      Z = Z + x_t
      K_ZZ = kernel(all_params,kernel_state,Z,Z)
      K_XZ = kernel(all_params,kernel_state,X,Z)
      
      L =  update_cholesky(K_XX,L_XX,K_XX_inv,K_XZ,K_ZZ,) ## Lower factor of (K_XZ_XZ)^-1
      K_xZ_dx = jax.jacrev(kernel.__call__,argnums=2)(all_params,kernel_state,x_t,Z).reshape(-1,d) ##[n,d]
      K_xXZ_dx = jnp.vstack([K_xX_dx,K_xZ_dx])

      factor1 = jax.scipy.linalg.cho_solve((L,False),K_xXZ_dx) ### K_XZ_XZ^-1 @ K_xXZ_dx
      K_xx_dx2 = -jax.hessian(kernel.__call__,argnums=2)(all_params,kernel_state,x_t,x_t).squeeze() ###watch out for -
      covar_xx_condXZ = K_xx_dx2 - K_xXZ_dx.T @ factor1 ### they call it  "covar_xstar_xstar_condx"
      
      ### In case you need to recover GIBO
      #info = jnp.trace(covar_xx_condXZ)

      ########################
      
      factor2 = jax.scipy.linalg.solve(K_XX,K_XZ,assume_a="pos")
      sigma_Z = K_ZZ - K_XZ.T @ factor2 ### This is Sigma_Z the covariance of observations y_Z
      covar_xx_condZ = K_xZ_dx.T - K_xX_dx.T @ factor2 ### Sigma_XZ 
      L_Z = cholesky(sigma_Z,lower=True) ### Lower factor i.e L@L.T = A


      A = jax.scipy.linalg.solve_triangular(L_Z,covar_xx_condZ.T,lower=True).T ###this one might cause error
      b1 = mean_d @ solve(covar_xx_condXZ,mean_d.T,assume_a="pos") 
      b2 = jnp.trace(A.T @ solve(covar_xx_condXZ,A,assume_a="pos"))
      info = (b1 + b2).squeeze()
      
      return -info
    

def _posterior_derivative(params,
                            mean_state,kernel_state,
                            x,X,y,
                            mean:Callable,kernel:Callable):


    MX = mean(params,mean_state,X)
    K_XX = kernel(params,kernel_state,X,X) 
    d = X.shape[-1]
    
    Mx_dx = jax.grad(mean.__call__,argnums=2)(params,mean_state,x).reshape(-1,1)
    ###TODO:avoid reshape and use proper stuff
    KxX_dx = jax.jacrev(kernel.__call__,argnums=2)(params,kernel_state,x,X).squeeze().reshape(-1,d) ##[n,d]
    ### The computed hessian has wrong sign so we correct it manually
    K_xx_dx2 = -jax.hessian(kernel.__call__,argnums=2)(params,kernel_state,x,x).squeeze()
    y_d = y.reshape(-1,1)- MX.reshape(-1,1) 

    mean_d = Mx_dx + KxX_dx.T @ jax.scipy.linalg.solve(K_XX,y_d,assume_a="pos")
    variance_d = K_xx_dx2 - KxX_dx.T @ jax.scipy.linalg.solve(K_XX,KxX_dx,assume_a="pos")
    mean_d = mean_d.T
    
  
    return mean_d, variance_d

@partial(jax.jit,static_argnames=["mean","kernel","_posterior_derivative"])
def info_factors(params,
                   mean_state,kernel_state,
                   x,X,y,
                   mean:Callable,kernel:Callable,_posterior_derivative:Callable):
    
    
  
    ### Avoid recalculations in the loop
    d = X.shape[-1]
    K_XX = kernel(params,kernel_state,X,X)
    L_XX = cholesky(K_XX,lower=True).T ## upper factor i.e L.T@L = A
    K_xX_dx = jax.jacrev(kernel.__call__,argnums=2)(params,kernel_state,x,X).squeeze().reshape(-1,d)
    K_XX_inv = jnp.linalg.inv(K_XX)
    mean_d,_ = _posterior_derivative(
        params,
        mean_state,kernel_state,
        x,X,y)
    
    return K_XX,K_xX_dx,K_XX_inv,L_XX,mean_d
  
  

def generate_acq_pts(x,rng,
                     m,d,sigma,):

    pos_noise = jax.random.normal(rng, (m,d))
    all_noise = pos_noise
    Z =  sigma * all_noise
    Z_flat = Z.reshape(-1)
    Z_flat = jnp.clip(Z_flat,a_min=-0.1,a_max=0.1)
    
    return Z_flat
  
def _get_query_pts(rng,
                      params,
                      mean_state,kernel_state,
                      x,X,y,
                      K_XX,K_xX_dx,K_XX_inv,L_XX,mean_d,
                      m:int,d:int,sigma:float,
                      info:Callable,info_factors:Callable,generate_acq_pts:Callable,optim_run:Callable,
                      ):


      Z_init= generate_acq_pts(x,rng=rng,sigma=sigma,m=m)
      
      Z_init_info = info(Z_init,
                                all_params = params,
                                mean_state=mean_state,kernel_state=kernel_state,
                                x_t=x,X=X,y=y,
                                K_XX = K_XX,L_XX=L_XX,K_XX_inv=K_XX_inv,K_xX_dx=K_xX_dx,mean_d=mean_d)
      rslt = optim_run(Z_init,  
                                bounds = (-0.1*jnp.ones_like(Z_init),0.1*jnp.ones_like(Z_init)),
                                all_params = params,
                                mean_state=mean_state,kernel_state=kernel_state,
                                x_t=x,X=X,y=y,
                                K_XX = K_XX,L_XX=L_XX,K_XX_inv=K_XX_inv,K_xX_dx=K_xX_dx,mean_d=mean_d)
      
      Z_optim= rslt.params
      Z_optim_info = rslt.state.value
      Z_optim = Z_optim.reshape(m,d)+x
      info_gain = -(Z_optim_info-Z_init_info)
      
      return Z_optim,Z_optim_info,info_gain

  
  
  

class GPD(GP):


  def __init__(self,mean,kernel,
               m,d,sigma):
    
    super().__init__(mean,kernel)

    self.m,self.d,self.sigma = m,d,sigma
    
    self.info = partial(info,mean=self.mean.__call__,kernel=self.kernel.__call__)
    self._posterior_derivative = partial(_posterior_derivative,mean=self.mean.__call__,kernel=self.kernel.__call__)
    self.info_factors = partial(info_factors,mean=self.mean.__call__,kernel=self.kernel.__call__,_posterior_derivative=self._posterior_derivative)
  
    
    self.acq_optim = jaxopt.LBFGSB(fun=self.info)
    
    self.generate_acq_pts = partial(generate_acq_pts,d=self.d)
    
    self._get_query_pts = partial(_get_query_pts,
                                  d=self.d,sigma=self.sigma,
                                  info_factors=self.info_factors,
                                  info=self.info,generate_acq_pts=self.generate_acq_pts,
                                  optim_run = self.acq_optim.run,
                                  )
    
    self.get_query_pts_vmap = jax.jit(jax.vmap(self._get_query_pts,in_axes=(0,         
                                                                    None,
                                                                    None,None,
                                                                    None,None,None,
                                                                    None,None,None,None,None, ## K_XX ...
                                                                    None #m
                                                                    )),static_argnums=12)                                    
    
  def posterior_derivative(self,x):
    
    mean_d, variance_d =  self._posterior_derivative(
                                  self.params,
                                  self.mean_state,self.kernel_state,
                                  x,self.X,self.y)
    
    return mean_d,variance_d



    
  def get_query_pts(self,rng,x,m):

    rngs = jax.random.split(rng,32)
    
    
    K_XX,K_xX_dx,K_XX_inv,L_XX,mean_d= self.info_factors(self.params,
                                                           self.mean_state,self.kernel_state,
                                                            x,self.X,self.y)
      
    Z_b,Z_info_b,info_gain_b = self.get_query_pts_vmap(rngs,
                                                      self.params,self.mean_state,self.kernel_state,
                                                        x,self.X,self.y,
                                                        K_XX,K_xX_dx,K_XX_inv,L_XX,mean_d,
                                                        m)
    
    Z_max,Z_info_max,info_gain_max = fetch_best(Z_b,Z_info_b,info_gain_b)

    
    return Z_max,Z_info_max,info_gain_max
  

  