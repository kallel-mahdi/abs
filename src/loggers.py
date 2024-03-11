import wandb
import jax
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotly.express as px


class Logger(object):
    
    
    def __init__(self,mlp,param_reshaper,num_rollouts,num_rollouts_center,
                 use_wandb=True,project_name=None,run_name=None,hypers_cfg=None,
                 *args,**kwargs):
    
    
        apply_many_states = jax.vmap(mlp, 
                                    in_axes=(None,None,0),out_axes=1)
        self.run_params = jax.vmap(apply_many_states, 
                                    in_axes=(param_reshaper.vmap_dict,None,None))#run_params(params,obs_params,states) ==> (b_params,act_dim,b_states)
        
        
        if use_wandb : 
            
            os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
            self.wb_run = wandb.init(project=project_name,name=run_name,config=hypers_cfg,reinit=True)
            
            
        
        self.n_episodes = 0
        self.n_transitions = 0
        self.max_return = jnp.zeros((1,1))
        self.disc_max_return = jnp.zeros((1,1))
        self.num_rollouts_center = num_rollouts_center
        self.use_wandb = use_wandb
        self.param_reshaper = param_reshaper
        self.old_ls = None

    
    def to_numpy(self,dictionary):
        
        return {key:np.array(value) for key,value in dictionary.items() }
    
        
    def log(self,dictionary):
            
           
        dictionary.update({"n_episodes":self.n_episodes,"n_transitions":self.n_transitions})

        
        commit = "return_grad" in dictionary.keys()
        dictionary = self.to_numpy(dictionary)
        
        if self.use_wandb :
 
            wandb.log(dictionary,commit=commit)
        

    
    def log_sample_efficiency(self,n_transitions,n_episodes):
        
        self.n_episodes += n_episodes
        self.n_transitions += n_transitions
        
        
    
    def log_perf(self,y_acq,y_t,test) :
        
        
        return_step = jnp.vstack([y_t.reshape(-1,1),y_acq])
        
        if test :
            
            
            
            tmp = jnp.vstack([self.max_return.reshape(-1,1),return_step])
            self.max_return = jnp.max(tmp)
            
            self.log({
                
                "return_acq":y_acq.mean(),
                "return_acq_max":y_acq.max(),
                "return_grad":y_t.mean(),
                "return_step" : return_step.mean(),
                "absolute_maximum" : self.max_return,
                
            })
            
        else :
            
            
            tmp = jnp.vstack([self.disc_max_return.reshape(-1,1),return_step])
            self.disc_max_return = jnp.max(tmp)
            
        
            self.log({
                
                "disc_return_acq":y_acq.mean(),
                "disc_return_max":y_acq.max(),
                "disc_return_grad":y_t.mean(),
                "disc_return_step" : return_step.mean(),
                "absolute_disc_maximum" : self.disc_max_return
            })

    
    def log_dist(self,x1,x2,obs_params,states,masks,is_grad):
        

        rng = jax.random.PRNGKey(1)
        
        states = states.reshape((-1,*states.shape[2:]))
        masks = masks.reshape((-1,*masks.shape[2:]))

        
    
        x1_r = self.param_reshaper.reshape(x1)
        x2_r = self.param_reshaper.reshape(x2)
        
        
        a1 = self.run_params(x1_r,obs_params,states)
        a2 = self.run_params(x2_r,obs_params,states) ### [N_params,A_dim,N_states]


        a1*=masks.squeeze()
        a2*=masks.squeeze()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

        
        param_distance_to_local = jax.numpy.abs(x1-x2).mean()
        action_distance_to_local = jax.numpy.abs(a1-a2).mean(axis=(0,1)).sum(axis=-1)/masks.sum()
        

        
        if is_grad : 

            self.log({
                    "param_distance_grad":param_distance_to_local,
                    "action_distance_grad":action_distance_to_local,
                    })


        else : 
        
            self.log({
                "param_distance_to_local":param_distance_to_local,
                "action_distance_to_local":action_distance_to_local,
                    })
            
        
        avg_dist_action = jax.numpy.abs(a1-a2).mean(axis=0).sum(axis=-1)/masks.sum()

        return avg_dist_action
        
        
            
    
    
    def log_hypers(self,outputscale,lengthscales,noise,constant=0,
                   avg_dist_param=None,avg_dist_grad=None):
        
        
        self.log( {
            
            "covar_lengthscale median": jnp.median(lengthscales),
            "covar_lengthscale mean": lengthscales.mean(),
            "covar_lengthscale std":  lengthscales.std(),
            "covar_output_scale" : outputscale,
            "mean_const":constant,
            "noise": noise,  
            
            })

    def log_fitting_metrics(self,y,y_pred,local=False,real=False):
        
        y,y_pred = np.array(y),np.array(y_pred)
        y_pred = y_pred.reshape(-1,1)
        a2 = ((y-y_pred)**2).sum()+1e-4
        b2=((y-y.mean())**2).sum()
        R2 = 1-(a2/b2)
        

    
        dict = {"R2":float(R2)}

        if local:
            
            dict = {f'{k}local': v for k, v in dict.items()}
        
            if real:
                


                y_diff= (y-y_pred)

                dict = {f'{k}_real': v for k, v in dict.items()}

                dict.update({"error_std":y_diff.std(),"error_avg":jnp.abs(y-y_pred).mean(),"error_bias":(y_pred-y).mean()})
               
                sort_idx = np.argsort(y.squeeze())
                y = y[sort_idx]
                y_pred = y_pred[sort_idx]
                x_axis = np.arange(len(y_pred))
                
                x = np.hstack([x_axis,x_axis]).squeeze()
                y = np.vstack([y,y_pred]).squeeze()
                real_labels = np.array(['real' for i in range(len(x_axis))])
                pred_labels = np.array(['pred' for i in range(len(x_axis))])
                label = np.hstack([real_labels,pred_labels])
                fig = px.bar(x=x,y=y,color=label,
                            barmode='group',
                            height=400
                            )

                wandb.log({"Predicted vs Real values (sorted)": fig},commit=False)


        self.log(dict)
    
    
    def log_R2s(self,R2s):
        
        x = np.arange(len(R2s))
        fig = px.bar(x=x,y=R2s,height=400)
        wandb.log({"R2 factor of each ensemble": fig},commit=False)

        
        
     


    
    
        
        
        
        
        
        
