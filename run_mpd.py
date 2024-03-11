
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import os
import time
import pickle
import wandb
import argparse
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from src.utils import CodeTimer




def none_or_str(value):
    if value == 'None':
        return None
    return value

# Set env variables
os.environ["WANDB_API_KEY"]="YOUR_API_KEY"
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

##############################
parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=42) 
parser.add_argument('--task_name',type=str,default="swimmer") 
parser.add_argument('--project_name',type=str,default="WANDB_PROJECT_NAME") 
parser.add_argument('--algo_name',type=str,default="abs") ## ["abs","mpd"]
parser.add_argument('--learning_rate',type=float,default=None)     
parser.add_argument('--lengthscale_bound',type=float,nargs='+',default=None) 
parser.add_argument('--reset_critic',type=bool,default=True) 
parser.add_argument('--aggregation',type=str,default="mean")## ["mean","softmax"]
args = parser.parse_args()



np.random.seed(40)
seeds = list(np.random.randint(0,1e6,5))
cfg = itertools.product(seeds,[args.task_name],[args.project_name],[args.algo_name],
                        [args.learning_rate],[args.lengthscale_bound],
                        [args.reset_critic],[args.aggregation])


# cfg = itertools.product([[args.seed]],[args.task_name],[args.project_name],[args.algo_name],
#                         [args.learning_rate],[args.lengthscale_bound],
#                         [args.reset_critic],[args.aggregation])

def train(config):    

        import os
        import random
        import numpy
        import jax
        import jax.numpy as jnp
        from jax.config import config as cfg
        
        seed = config[0]
        random.seed(seed)
        np.random.seed(seed)

        cfg.update("jax_enable_x64", True)
        cfg.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)        
    
        from src.setup import get_mpd_cfgs,setup_mpd,setup_ars
        from src.fitness import fitness_acq,fitness_center,update_norm

        configs = get_mpd_cfgs(*config)
        
        mpd,evaluator,logger,param_reshaper,rng = setup_mpd(*configs)

        num_rollouts,num_rollouts_center = mpd.num_rollouts,mpd.num_rollouts_center  
        normalizer_params = evaluator.normalizer_params
        
        
        if config[3]in ["abs","ddpgn"]:

                    mpd.gp.mean_state = mpd.gp.mean_state.replace(obs_params=normalizer_params)

        ascent_prob = 1.
        
        for i in range(mpd.num_learning_steps):
        
            if i>0 : 
                
                ### Take gradient step
                x_old = mpd.x
                x_t,ascent_prob = mpd.ask_local()
                normalizer_params = update_norm(normalizer_params,obs_acq,mask_acq!=0)
                if config[3] in ["abs","ddpgn"]:
                    mpd.gp.mean_state = mpd.gp.mean_state.replace(obs_params=normalizer_params)
                

            else : 
                
                x_t = jnp.zeros((1,mpd.n_dims))
                x_old = x_t
                
                
            y_t,noise_t,obs_t, mask_t,transitions_t= fitness_center(
                                                    evaluator,
                                                    param_reshaper,x_t,normalizer_params,
                                                    num_rollouts_center)
        
            mpd.tell_local(rng,x_t,y_t,noise_t,
                            obs_t,mask_t,tuple(transitions_t),obs_params=normalizer_params,
                            )
            
            
            
            if i == 0 :
                
                x_acq = mpd.gp.sigma * jax.random.normal(rng,(mpd.n_max-1,mpd.n_dims))
                y_acq,obs_acq,mask_acq,transitions_acq,noise_acq = fitness_acq(evaluator,
                                                                        param_reshaper,x_acq,normalizer_params,num_rollouts=mpd.num_rollouts)
                        
                mpd.tell_neighbours(rng,x_acq,y_acq,noise_acq,
                                    obs_acq,mask_acq,tuple(transitions_acq))
                
                
                
            elif i > 0 :
                
                x_acq_tmp,y_acq_tmp,obs_acq_tmp,mask_acq_tmp = [],[],[],[]
                    
                for _ in range(mpd.n_info//mpd.n_parallel):

                        rng,_ =  jax.random.split(rng)
                        
                        x_acq,x_acq_info,info_gain = mpd.ask_neighbours(rng,mpd.n_parallel)
                        
                        y_acq,obs_acq,mask_acq,transitions_acq,noise_acq = fitness_acq(evaluator,
                                                                        param_reshaper,x_acq,normalizer_params,num_rollouts=mpd.num_rollouts)
                        
                        mpd.tell_neighbours(rng,x_acq,y_acq,noise_acq,
                                            obs_acq,mask_acq,tuple(transitions_acq))
                        
                        
                        x_acq_tmp.append(x_acq)
                        y_acq_tmp.append(y_acq)
                        obs_acq_tmp.append(obs_acq)
                        mask_acq_tmp.append(mask_acq)
                    
                x_acq = jnp.vstack(x_acq_tmp)
                y_acq = jnp.vstack(y_acq_tmp)
                obs_acq = jnp.vstack(obs_acq_tmp)
                mask_acq = jnp.vstack(mask_acq_tmp)
                
                
            ############# LOG METRICS ETC #################    

            y_acq_real,_,_,_,_ = fitness_acq(evaluator,
                                        param_reshaper,x_acq,normalizer_params,num_rollouts=5,eval=False)
            
            y_t_real,_, _, _,_ = fitness_center(evaluator,
                                            param_reshaper,x_t,normalizer_params,num_rollouts=5,eval=False)
            
            y_t_undisc,_, _, _,_ = fitness_center(evaluator,
                                            param_reshaper,x_t,normalizer_params,num_rollouts=5,eval=True)
            
            y_acq_undisc,_,_,_,_ = fitness_acq(evaluator,
                                        param_reshaper,x_acq,normalizer_params,num_rollouts=1,eval=True)
            

            

            
            avg_dist_param = logger.log_dist(mpd.x,x_acq,normalizer_params,obs_t,mask_t,is_grad=False)
            avg_dist_grad = logger.log_dist(x_old,x_t,normalizer_params,obs_t,mask_t,is_grad=True)
            
            y_pred = mpd.gp.mean.__call__(mpd.gp.params,mpd.gp.mean_state,mpd.gp.X)
            y_pred_local = mpd.gp.mean.__call__(mpd.gp.params,mpd.gp.mean_state,x_acq)

            logger.log_fitting_metrics(y=mpd.gp.y,y_pred=y_pred,local=False)
            logger.log_fitting_metrics(y=y_acq_real,y_pred=y_pred_local,local=True,real=True)
            logger.log_fitting_metrics(y=y_acq,y_pred=y_pred_local,local=True,real=False)

            logger.log({
                        
                        "noise_center":mpd.gp.noise_t,
                        "signal_noise":mpd.gp.signal_noise,
                        "snr":mpd.gp.snr,
                        "signal_variance":mpd.gp.signal_variance,
                        "ascent_prob":ascent_prob,
                        }) 
            
                   
            if config[3]in ["abs","ddpgn"]:
                
                logger.log({"R2_validation":jnp.max(mpd.gp.mean.R2_history)}) 
                logger.log_R2s(mpd.gp.mean.R2_history)
                      
            
            
            logger.log_perf(y_acq_undisc,y_t_undisc,test=True)
            logger.log_perf(y_acq_real,y_t_real,test=False)
            logger.log_hypers(avg_dist_param=avg_dist_param,avg_dist_grad=avg_dist_grad,**mpd.gp.params)
            logger.log_sample_efficiency((mask_acq.sum()+mask_t.sum()),(mpd.num_rollouts*mpd.n_info+mpd.num_rollouts_center))


            print("y_acq_real",y_acq_real.mean(),"y_t_real",y_t_real)
            
            rng,_ =  jax.random.split(rng)
            
           
        logger.wb_run.finish()



    
if __name__ == "__main__":

    wandb.require("service")
    wandb.setup()

    for c in cfg : 
    
        train(c)

    
    wandb.finish()

        
    


