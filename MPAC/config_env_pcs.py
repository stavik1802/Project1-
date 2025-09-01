# args for pcs env

from energy_net.market.pricing_policy import PricingPolicy
from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType

def get_env_kwargs():
    return {
        "action": {
            "consumption_action": {"enabled": False, "min": -10.0, "max": 10.0},
            "production_action": {"enabled": False, "min": -10.0, "max": 10.0},
            "multi_action": False,
        },
        "observation_space": {
            "battery_level": {"min": "from_battery_config", "max": "from_battery_config"},
            "time": {"min": 0.0, "max": 1.0},
            "iso_buy_price": {"min": 0.0, "max": 1000.0},
            "iso_sell_price": {"min": 0.0, "max": 1000.0},
        },
        "battery": {
            "dynamic_type": "model_based",
            "model_type": "deterministic_battery",
            "model_parameters": {
                "charge_efficiency": 1.0,
                "discharge_efficiency": 1.0,
                "charge_rate_max": 10.0,
                "discharge_rate_max": 10.0,
                "init": 40.0,
                "lifetime_constant": 100.0,
                "max": 100.0,
                "min": 0.0,
            },
        },
        "consumption_unit": {
            "dynamic_type": "model_based",
            "model_type": "deterministic_consumption",
            "model_parameters": {
                "consumption_capacity": 0.0,
                "peak_consumption1": 0.0,
                "peak_consumption2": 0.0,
                "peak_time1": 0.0,
                "peak_time2": 0.0,
                "width1": 1.0,
                "width2": 1.0,
            },
        },
        "production_unit": {
            "dynamic_type": "model_based",
            "model_type": "deterministic_production",
            "model_parameters": {
                "peak_production": 0.0,
                "peak_time": 0.0,
                "production_capacity": 0.0,
                "width": 1.0,
            },
        },
        "dispatch_cost": {
            "thresholds": [
                {"level": 100.0, "rate": 5.0},
                {"level": 170.0, "rate": 7.0},
                {"level": "inf", "rate": 8.0},
            ],
        },
        "default_iso_params": {
            "quadratic": {
                "buy_a": 1.0,
                "buy_b": 2.0,
                "buy_c": 5.0,
            }
        },
    }

def get_env_setup_kwargs():
    return {
        "pricing_policy": PricingPolicy.ONLINE,
        "cost_type": CostType.CONSTANT,
        "demand_pattern": DemandPattern.SINUSOIDAL,
    }