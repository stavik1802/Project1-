from MPAC.MPAC_wrapper import MPACWrapper
import torch
import numpy as np

# Path to the saved model zip
model_path = "energy-net-zoo/logs/iso/ppo/run_1/mpac/ISO-RLZoo-v0_36/mpac_model.zip"

# Load the model using the wrapper's classmethod
policy = MPACWrapper.load(model_path)

# Create dummy observation
dummy_obs = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
dummy_obs_tensor = torch.tensor(dummy_obs, dtype=torch.float32)

# Predict using the loaded policy
action, _ = policy.predict(dummy_obs_tensor)
print("Predicted action:", action)
