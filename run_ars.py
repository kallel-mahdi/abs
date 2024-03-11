import warnings
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import os
import time
import wandb
import argparse
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from src.utils import CodeTimer


os.environ["WANDB_API_KEY"]="YOUR_API_KEY"
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

##############################
parser = argparse.ArgumentParser()
parser.add_argument('--task_name',type=str,default="walker2d") 
parser.add_argument('--project_name',type=str,default="WANDB_PROJECT_NAME") 
parser.add_argument('--lr',type=float,default=None) 
args = parser.parse_args()

##############################
#PROJECT_NAME = args.project_name
np.random.seed(42)
seeds = list(np.random.randint(0,1e6,5))
ranks = list(range(len(seeds)))
##############################
learning_rate = [None] ##0.0025

cfg = itertools.product(seeds,[args.task_name],[args.project_name],learning_rate)


def train(config):    

    print("confiiig",config)

    import os
    import random
    import numpy
    import jax
    import jax.numpy as jnp
    from jax.config import config as cfg
    
    seed = config[0]
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    cfg.update("jax_enable_x64", True)
    cfg.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

  
    from src.setup import get_ars_cfgs,setup_ars
    from src.fitness import fitness_acq,fitness_center,update_norm    

    print('CONFIIIIIG',*config)
 
    configs = get_ars_cfgs(*config)
    
    algo,evaluator,logger,param_reshaper,rng = setup_ars(*configs)
    
    num_rollouts,num_rollouts_center = algo.num_rollouts,algo.num_rollouts_center  
    normalizer_params = evaluator.normalizer_params
    
    ascent_prob = 1.
    for i in range(algo.num_learning_steps):
    

        rng,_ =  jax.random.split(rng)
        
        
        if i>0 : 
            
          
            normalizer_params = update_norm(normalizer_params,obs_acq,mask_acq!=0)
            ### Take gradient step
            x_old = algo.x
            x_t,ascent_prob = algo.ask_local()
            

        else : 
            
            x_t = jnp.zeros((1,algo.n_dims))
            x_old = x_t
            
            
        
        y_t,noise_t,obs_t, mask_t,transitions_t= fitness_center(
                                                evaluator,
                                                param_reshaper,x_t,normalizer_params,
                                                num_rollouts_center)
    
        
        algo.tell_local(rng,x_t,y_t,noise_t,obs_t,mask_t,tuple(transitions_t),obs_params=normalizer_params)
        
           
            
        x_acq,x_acq_info,info_gain = algo.ask_neighbours(rng)

        y_acq,obs_acq,mask_acq,transitions_acq,_ = fitness_acq(evaluator,
                                                        param_reshaper,x_acq,normalizer_params)
        
        algo.tell_neighbours(rng,x_acq,y_acq,obs_acq,mask_acq,tuple(transitions_acq))
          
        ############# LOG METRICS ETC #################    

        y_acq_real,_,_,_,_ = fitness_acq(evaluator,
                                    param_reshaper,x_acq,normalizer_params,num_rollouts=5,eval=False)
        
        y_t_real,_, _, _,_ = fitness_center(evaluator,
                                        param_reshaper,x_t,normalizer_params,num_rollouts=5,eval=False)
        
        y_t_undisc,_, _, _,_ = fitness_center(evaluator,
                                        param_reshaper,x_t,normalizer_params,num_rollouts=5,eval=True)
        
        y_acq_undisc,_,_,_,_ = fitness_acq(evaluator,
                                    param_reshaper,x_acq,normalizer_params,num_rollouts=1,eval=True)
        

        
        
        logger.log_dist(algo.x,x_acq,normalizer_params,obs_t,mask_t,is_grad=False)
        logger.log_dist(x_old,x_t,normalizer_params,obs_t,mask_t,is_grad=True)

        logger.log_perf(y_acq_undisc,y_t_undisc,test=True)
        logger.log_perf(y_acq_real,y_t_real,test=False)
        
        logger.log_sample_efficiency((mask_acq.sum()),(algo.num_rollouts*algo.n_info))


        print("y_acq_real",y_acq_real.mean(),"y_t_real",y_t_real)
            
        
        
    
    logger.wb_run.finish()
    
   

    
if __name__ == "__main__":

    wandb.require("service")
    wandb.setup()

    for c in cfg : 
    
        train(c)

    
    wandb.finish()

    
  


