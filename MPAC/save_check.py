import os
import numpy as np
from pathlib import Path
import torch

# Dummy actor, critic, and normalizer classes for test
class DummyNet:
    def __init__(self):
        self.params = [torch.nn.Parameter(torch.randn(3, 3)), torch.nn.Parameter(torch.randn(3))]

    def get_weights(self):
        return [p.detach().cpu().numpy() for p in self.params]

    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            p.data.copy_(torch.tensor(w))


class DummyNormalizer:
    def get_rms_stats(self):
        return {"mean": np.zeros(3), "std": np.ones(3)}

    def set_rms_stats(self, stats):
        print("Restored normalizer stats:", stats)


class DummyMPACModel:
    def __init__(self):
        self.actor = DummyNet()
        self.critic = DummyNet()
        self.normalizer = DummyNormalizer()


# MPACWrapper mock
class MPACWrapper:
    def __init__(self, env=None, inputs_dict=None):
        self.env = env
        self.inputs_dict = inputs_dict or {"dummy": "config"}
        self.model = DummyMPACModel()

    def save(self, path):
        import tempfile, zipfile, cloudpickle
        path = Path(path)
        if path.is_dir():
            path = path / "mpac_model.zip"
        if not path.name.endswith(".zip"):
            path = path.with_suffix(".zip")

        os.makedirs(path.parent, exist_ok=True)

        with zipfile.ZipFile(path, "w") as archive:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_data:
                cloudpickle.dump({
                    "model_class": "MPACWrapper",
                    "inputs_dict": self.inputs_dict
                }, tmp_data)
                tmp_data_path = tmp_data.name
            archive.write(tmp_data_path, arcname="data.pkl")
            os.remove(tmp_data_path)

            with tempfile.NamedTemporaryFile(delete=False) as tmp_weights:
                torch.save({
                    "actor_weights": self.model.actor.get_weights(),
                    "critic_weights": self.model.critic.get_weights(),
                    "normalizer": self.model.normalizer.get_rms_stats()
                }, tmp_weights)
                tmp_weights_path = tmp_weights.name
            archive.write(tmp_weights_path, arcname="policy.pth")
            os.remove(tmp_weights_path)

    @classmethod
    def load(cls, path):
        import zipfile, cloudpickle, torch
        path = Path(path)
        if path.is_dir():
            path = path / "mpac_model.zip"
        if not path.name.endswith(".zip"):
            path = path.with_suffix(".zip")

        with zipfile.ZipFile(path, "r") as archive:
            with archive.open("data.pkl") as f:
                data = cloudpickle.load(f)
            with archive.open("policy.pth") as f:
                weights = torch.load(f, map_location="cpu")

        model = cls(env=None, inputs_dict=data["inputs_dict"])
        model.model.actor.set_weights(weights["actor_weights"])
        model.model.critic.set_weights(weights["critic_weights"])
        model.model.normalizer.set_rms_stats(weights["normalizer"])
        return model


# Test it
if __name__ == "__main__":
    save_path = "tmp_test/mpac_model.zip"

    print("Saving model...")
    wrapper = MPACWrapper()
    wrapper.save(save_path)

    print("\nLoading model...")
    loaded = MPACWrapper.load(save_path)
    print("Load successful. Inputs dict:", loaded.inputs_dict)
