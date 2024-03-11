from pathlib import Path
import yaml
from src.utils import ParameterReshaper  # , NetworkMapper
from src.fitness import BraxFitness
from src.ars import ARS
from src.abs import MPD
from src.ddpgn import DDPGN
from src.gpd import GPD

from src.kernels import *
from src.loggers import Logger
from src.means.agent import Agent
from src.means.mean import AdvantageMean, ConstantMean
from src.means.utils import ReplayBuffer
from src.networks import *
import src.utils

class UniformPrior:
        
        def __init__(self,a,b) -> None:
                self.a = a
                self.b = b
                    
                
def get_mpd_cfgs(seed,task_name,project_name,
             algo_name,
             learning_rate=None,lengthscale_bound=None,
             reset_critic=None,aggregation=None,
             ):


    cfg_path = "./configs/"+algo_name+"/"+task_name+".yaml"
    dict = yaml.safe_load(Path(cfg_path).read_text())

    env_cfg = dict.get("env_cfg",{})
    algo_cfg = dict.get("algo_cfg",{})
    mean_cfg = dict.get("mean_cfg",{})
    kernel_cfg = dict.get("kernel_cfg",{})
    policy_cfg = dict.get("policy_cfg",{})
    critic_cfg = dict.get("critic_cfg",{})
    logger_cfg = dict.get("logger_cfg",{})
    env_cfg["seed"]= seed
    
    if learning_rate is not None :
        algo_cfg["learning_rate"]=learning_rate
        
    if lengthscale_bound is not None :
        kernel_cfg["lengthscale_bound"] = lengthscale_bound
    
    if reset_critic is not None :
        mean_cfg["reset_critic"]=reset_critic
    
    if aggregation is not None :
        mean_cfg["aggregation"]=aggregation
           
        
    if project_name is not None : 
        
        logger_cfg["project_name"] = project_name
    
    
    logger_cfg["hypers_cfg"] = {**{"env_discount":env_cfg["discount"]},**env_cfg,**algo_cfg,**mean_cfg,**kernel_cfg,**policy_cfg,**critic_cfg}
    
    mean_cfg["algo_name"]= algo_name
    kernel_cfg["lengthscale_prior"] = UniformPrior(*kernel_cfg.pop("lengthscale_bound"))
    critic_cfg["discount"] = env_cfg["discount"]
    ##################################

    logger_cfg["run_name"] = mean_cfg["algo_name"]+"_"+str(algo_cfg["learning_rate"]) +"_"+ str(1 *env_cfg["normalize_obs"]) +"_"+ str(seed)
    logger_cfg["num_rollouts_center"] = algo_cfg["num_rollouts_center"]
            

    return seed,env_cfg,policy_cfg,mean_cfg,kernel_cfg,algo_cfg,critic_cfg,logger_cfg


def get_ars_cfgs(seed,task_name,project_name,
             learning_rate=None,
             ):


    cfg_path = "./configs/ars/"+task_name+".yaml"
    dict = yaml.safe_load(Path(cfg_path).read_text())

    env_cfg = dict["env_cfg"]
    policy_cfg = dict["policy_cfg"]
    env_cfg["seed"]= seed
    algo_cfg = dict["algo_cfg"]
    logger_cfg = dict["logger_cfg"]
    
    if learning_rate is not None :
        algo_cfg["learning_rate"]=learning_rate
        
    if project_name is not None : 

        logger_cfg["project_name"] = project_name
    
    ###########################

    logger_cfg["run_name"] = "ars_"+str(algo_cfg["learning_rate"]) +"_"+ str(seed)
    logger_cfg["hypers_cfg"] = {**{"env_discount":env_cfg["discount"]},**env_cfg,**algo_cfg,**policy_cfg}


    return seed,env_cfg,policy_cfg,algo_cfg,logger_cfg



def setup_mpd(seed,env_cfg,policy_cfg,mean_cfg,kernel_cfg,algo_cfg,critic_cfg,logger_cfg,use_wandb=True):

    rng = jax.random.PRNGKey(seed)
    evaluator = BraxFitness(**env_cfg,test=False)
    
    
    ###############################################
    # Create network and reshaper to transform flat params to DNN
    policy_cfg["num_output_units"] = evaluator.action_shape    
    actor_base_cls = partial(MLP,policy_cfg.pop("hidden_dims"))
    
    if env_cfg["env_name"] == "Pendulum-v1":
         
        network = PendulumPolicyNetwork(actor_base_cls,**policy_cfg)
    
    else :
        
        network = PolicyNetwork(actor_base_cls,**policy_cfg)

    normalize_obs = env_cfg["normalize_obs"]
    normalize_fn = src.utils.normalize if normalize_obs else   lambda x, y: x
    network = make_normalized_network(network,normalize_fn)
    ###############################################
    
    pholder = jnp.zeros((1, evaluator.input_shape[0]))
    params = network.init(rng,pholder)
    param_reshaper = ParameterReshaper(params,n_devices=1)
    evaluator.set_apply_fn(param_reshaper.vmap_dict, network.apply)
    d = int(param_reshaper.total_params)
    
    
    algo_name = mean_cfg.pop("algo_name")
    
    if algo_name =="mpd":
        
        mean = ConstantMean(UniformPrior(-1,1))
    
    elif algo_name in ["abs","ddpgn"] : 
    
       
        agent  = Agent(
                    state_dim=evaluator.input_shape[0],
                    action_dim=evaluator.action_shape,
                    actor_model=network,
                    actor_params=jax.numpy.zeros(d),
                    param_reshaper= param_reshaper,
                    normalize_obs=normalize_obs,
                    seed=seed,
                    **critic_cfg,
                    )

        mean = AdvantageMean(agent=agent,obs_params=evaluator.normalizer_params,
                            num_rollouts_center=algo_cfg["num_rollouts_center"],
                            num_rollouts=algo_cfg["num_rollouts"],
                            state_dim=evaluator.input_shape[0],
                            action_dim=evaluator.action_shape,
                            **mean_cfg,)
    
    #################
    
    d = int(param_reshaper.total_params)
    kernel = RBFKernel(**kernel_cfg,
                        ard_num_dims = d) 

    gp = GPD(mean,kernel,d=d,m=algo_cfg["n_info"],sigma=algo_cfg.pop("sigma"))
    
    if algo_name in ["abs","mpd"]:
        
        algo = MPD(**algo_cfg,
                    max_grad=kernel.u_b["lengthscales"],
                    gp=gp,x_sample=jnp.zeros((1,d)))
    
    if algo_name == "ddpgn":

        algo = DDPGN(**algo_cfg,
                    max_grad=kernel.u_b["lengthscales"],
                    gp=gp,x_sample=jnp.zeros((1,d)))
         
    logger = Logger(network.apply,param_reshaper,algo_cfg["num_rollouts"],
                        use_wandb=use_wandb,**logger_cfg)

    return algo,evaluator,logger,param_reshaper,rng

def setup_ars(seed,env_cfg,policy_cfg,algo_cfg,logger_cfg,use_wandb=True):
    
    
    rng = jax.random.PRNGKey(seed)
    evaluator = BraxFitness(**env_cfg,test=False)
    
    
    
    ###############################################
    policy_cfg["num_output_units"] = evaluator.action_shape    
    actor_base_cls = partial(MLP,policy_cfg.pop("hidden_dims"))
    
    
    if env_cfg["env_name"] == "Pendulum-v1":
         
        network = PendulumPolicyNetwork(actor_base_cls,**policy_cfg)
    
    else : 
        
        network = PolicyNetwork(actor_base_cls,**policy_cfg)
        
    normalize_obs = env_cfg["normalize_obs"]
    normalize_fn = src.utils.normalize if normalize_obs else   lambda x, y: x
    network = make_normalized_network(network,normalize_fn)
    ###############################################
    
    pholder = jnp.zeros((1, evaluator.input_shape[0]))
    params = network.init(rng,pholder)
    param_reshaper = ParameterReshaper(params,n_devices=1)
    evaluator.set_apply_fn(param_reshaper.vmap_dict, network.apply)
    d = int(param_reshaper.total_params)
    
    
    ars =ARS(**algo_cfg,n_dims=d)
    logger_cfg["num_rollouts_center"] = 0
    logger = Logger(network.apply,param_reshaper,algo_cfg["num_rollouts"],
                        use_wandb=use_wandb,**logger_cfg)
    
    return ars,evaluator,logger,param_reshaper,rng
