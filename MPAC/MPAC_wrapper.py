# mpac_wrapper.py
import torch
import pickle
import cloudpickle
from MPAC.train import train
import os
import torch
from MPAC.train_wrap import train

# mpac_wrapper.py -- now SB3 compatible
import os
import torch
import zipfile
import json
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from MPAC.train_wrap import train
from MPAC.algs.base_alg import BaseAlg
from MPAC.envs.wrappers.gymn_to_gym import GymnasiumToGymWrapper
from MPAC.envs.wrappers.energy_net_wrapper import EnergyNetWrapper
from pathlib import Path
import tempfile
from MPAC.actors import init_actor
from MPAC.common.normalizer import RunningNormalizers
class MPACWrapper:
    def __init__(self, env, inputs_dict, model_save_path=None):
        self.env = env
        self.inputs_dict = inputs_dict
        self.model = None
        self.model_save_path = model_save_path
        self.log_name = None
        self.env1 = env



    def learn(self, total_timesteps):

        self.inputs_dict["alg_kwargs"]["total_timesteps"] = total_timesteps
        #self.env = self.env.envs[0]
        self.env = GymnasiumToGymWrapper(self.env)
        self.model = train(self.inputs_dict,env=self.env,env_eval=self.env)  # Now returns a BaseAlg instance
        self.log_name = self.model.checkpoint_name

    def predict(self, obs, deterministic=True):
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        # You must define how prediction works with your actor
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        import inspect
        path = Path(path)
        if path.is_dir():
            path = path / "mpac_model.zip"
        if not path.name.endswith(".zip"):
            path = path.with_suffix(".zip")
        os.makedirs(path.parent, exist_ok=True)

        actor_class = self.model.actor.__class__
        actor_init_args = {
            k: v for k, v in vars(self.model.actor).items()
            if not k.startswith('_')  # Skip private/internal stuff
        }

        with zipfile.ZipFile(path, "w") as archive:
            data = {
                "model_class": "MPACWrapper",
                "env": self.env,
                "inputs_dict": self.inputs_dict,
                "actor_class": actor_class,
                "actor_init_args": actor_init_args,
            }
            with tempfile.NamedTemporaryFile(delete=False) as tmp_data:
                cloudpickle.dump(data, tmp_data)
                archive.write(tmp_data.name, arcname="data.pkl")
                os.remove(tmp_data.name)

            with tempfile.NamedTemporaryFile(delete=False) as tmp_weights:
                torch.save({
                    "actor_weights": self.model.actor.get_weights(),
                    "normalizer": self.model.normalizer.get_rms_stats(),
                }, tmp_weights)
                archive.write(tmp_weights.name, arcname="policy.pth")
                os.remove(tmp_weights.name)





    @classmethod
    def load(cls, path):
        import zipfile, cloudpickle, torch

        path = str(path)
        if not path.endswith(".zip"):
            path += ".zip"

        with zipfile.ZipFile(path, "r") as archive:
            data = cloudpickle.loads(archive.read("data.pkl"))
            weights = torch.load(archive.open("policy.pth"), map_location="cpu")
        env = data.get("env")  # will be <GymnasiumToGymWrapper instance>
        inputs_dict = data["inputs_dict"]
        # Use it to build actor
        from MPAC.actors import init_actor
        actor_init_args = data["actor_init_args"]
        actor_init_args.pop('device',None)
        actor = init_actor(env, **data["inputs_dict"]["actor_kwargs"])
        actor.set_weights(weights["actor_weights"])
        normalizer = None
        if "normalizer" in weights:
            from MPAC.common.normalizer import RunningNormalizers
            gamma = inputs_dict["alg_kwargs"]["gamma"]
            init_stats = inputs_dict["alg_kwargs"].get("init_rms_stats", None)
            s_dim = actor.s_dim
            a_dim = actor.a_dim
            r_dim = getattr(actor, "r_dim", 1)
            normalizer = RunningNormalizers(s_dim, a_dim, gamma, init_stats, r_dim)
            normalizer.set_rms_stats(weights["normalizer"])
            actor.s_rms = normalizer.s_rms

        return PolicyOnlyWrapper(actor, normalizer)



        # === Outside MPACWrapper ===
class PolicyOnlyWrapper:
    def __init__(self, actor, normalizer=None):
        self.actor = actor
        self.normalizer = normalizer
        if self.normalizer is not None:
            self.actor.s_rms = self.normalizer.s_rms

    def predict(self, obs, deterministic=True):
        import torch
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            action = self.actor.sample(obs_tensor.unsqueeze(0), deterministic=deterministic)
            action = action.cpu().numpy()

        return action, None


        # class MPACWrapper:
#     def __init__(self, env, inputs_dict, model_save_path=None):
#         self.env = env
#         self.inputs_dict = inputs_dict
#         self.model = None  # Will be set after `train()`
#         self.model_save_path = model_save_path or "mpac_model.pt"
#         self.log_name = None
#
#     def learn(self, total_timesteps):
#         self.inputs_dict["alg_kwargs"]["total_timesteps"] = total_timesteps
#         self.log_name = train(self.inputs_dict)
#         # After training, the model object is created inside `train()` and returned or accessible via log_name
#         # You need to expose and return the actual model (BaseAlg or similar) from `train()` to set here.
#
#     def predict(self, obs, deterministic=True):
#         if self.model is None:
#             raise RuntimeError("Model has not been trained or loaded.")
#         return self.model.predict(obs, deterministic=deterministic)
#
#     def save(self, path=None):
#         path = path or self.model_save_path
#         if self.model is None:
#             raise RuntimeError("Model must be trained before saving.")
#         weights = self.model._dump_stats()  # Should return dict of all weights/state
#         torch.save(weights, path)
#
#     def load(self, path=None):
#         path = path or self.model_save_path
#         weights = torch.load(path)
#         self.log_name = weights.get("log_name", None)
#
#         # Now reload all weights to the actor, critic, rob_net, etc.
#         # You must reconstruct model manually (via train_utils/init_alg.py etc.) before loading weights
#         self.model = self.rebuild_model()
#         self.model.load_weights(weights)
#
#     def rebuild_model(self):
#         """Rebuilds the model using inputs_dict."""
#         from MPAC.train import init_env, init_actor, init_critics, init_rob_net, init_alg
#         env, env_eval = init_env(**self.inputs_dict["env_kwargs"],
#                                  env_setup_kwargs=self.inputs_dict["env_setup_kwargs"])
#         actor = init_actor(env, **self.inputs_dict["actor_kwargs"])
#         critic, cost_critic = init_critics(env, **self.inputs_dict["critic_kwargs"],
#                                            multiobj_enable=self.inputs_dict["env_setup_kwargs"].get("multiobj_enable", False),
#                                            num_objectives=1)
#         rob_net = init_rob_net(env, **self.inputs_dict["rob_kwargs"],
#                                rob_setup_kwargs=self.inputs_dict["rob_setup_kwargs"],
#                                safety_kwargs=self.inputs_dict["safety_kwargs"])
#         alg = init_alg(self.inputs_dict["setup_kwargs"]["idx"], env, env_eval, actor, critic,
#                        cost_critic, rob_net, self.inputs_dict["alg_kwargs"],
#                        self.inputs_dict["safety_kwargs"], self.inputs_dict["rl_update_kwargs"])
#         return alg

# class MPACWrapper:
#     def __init__(self, env, inputs_dict, model_save_path=None):
#         self.env = env
#         self.inputs_dict = inputs_dict
#         self.model = None
#         self.model_save_path = model_save_path or "mpac_model.pt"
#         self.log_name = None
#
#
#
#     def learn(self, total_timesteps):
#         self.inputs_dict["alg_kwargs"]["total_timesteps"] = total_timesteps
#         self.log_name = train(self.inputs_dict)
#
#     def predict(self, obs, deterministic=True):
#         if self.model is None:
#             raise RuntimeError("Model has not been trained or loaded.")
#         # You must define how prediction works with your actor
#         return self.model.predict(obs, deterministic=deterministic)
#
#     def save(self, path=None):
#         path = path or self.model_save_path
#         with open(path, 'wb') as f:
#             pickle.dump(self.log_name, f)  # Or save actor weights if available
#
#     def load(self, path=None):
#         path = path or self.model_save_path
#         with open(path, 'rb') as f:
#             self.log_name = pickle.load(f)
#         # Load model weights here if your architecture supports it
