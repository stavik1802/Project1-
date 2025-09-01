# components/battery.py

from typing import Any, Dict, Optional
from energy_net.components.grid_entity import ElementaryGridEntity
from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.utils.logger import setup_logger  # Import the logger setup
from numpy.random import uniform,normal


class Battery(ElementaryGridEntity):
    """
    Battery component managing energy storage.
    """

    def __init__(self, dynamics: EnergyDynamics, config: Dict[str, Any], log_file: Optional[str] = 'logs/battery.log'):
        """
        Initializes the Battery with dynamics and configuration parameters.

        Args:
            dynamics (EnergyDynamics): The dynamics defining the battery's behavior.
            config (Dict[str, Any]): Configuration parameters for the battery.
            log_file (str, optional): Path to the Battery log file.

        Raises:
            AssertionError: If required configuration parameters are missing.
        """
        super().__init__(dynamics, log_file)
        
        # Set up logger
        self.logger = setup_logger('Battery', log_file)
        self.logger.info("Initializing Battery component.")
        #additional for perturbation
        # self.perturbation_enabled = config.get("perturbation_enabled", False)
        # self.perturbation_min = config.get("perturbation_min", 0.9)
        # self.perturbation_max = config.get("perturbation_max", 1.1)
        # Ensure that all required configuration parameters are provided
        required_params = [
            'min', 'max', 'charge_rate_max', 'discharge_rate_max',
            'charge_efficiency', 'discharge_efficiency', 'init'
        ]
        for param in required_params:
            assert param in config, f"Missing required parameter '{param}' in Battery configuration."

        self.energy_min: float = config['min']
        self.energy_max: float = config['max']
        self.charge_rate_max: float = config['charge_rate_max']
        self.discharge_rate_max: float = config['discharge_rate_max']
        self.charge_efficiency: float = config['charge_efficiency']
        self.discharge_efficiency: float = config['discharge_efficiency']
        self.initial_energy: float = config['init']
        self.energy_level: float = self.initial_energy
        self.energy_change: float = 0.0  # Initialize energy_change

        self.logger.info(f"Battery initialized with energy level: {self.energy_level} MWh")


    #change action to perturbed action if perturbation is enabled
    from numpy.random import uniform, normal

    def perform_action(self, action: float) -> None:
        """
        Performs charging or discharging based on the action by delegating to the dynamic.
        Applies per-step perturbations if enabled.
        """
        # # 1. Perturb action
        # if self.perturbation_enabled:
        #     scale = uniform(self.perturbation_min, self.perturbation_max)
        #     perturbed_action = action * scale
        #     self.logger.info(f"⚠️ Perturbed battery action: {action:.3f} → {perturbed_action:.3f} (scale={scale:.3f})")
        # else:
        #     perturbed_action = action

        previous_energy = self.energy_level


        self.energy_level = self.dynamics.get_value(
            time=self.current_time,
            action=action,
            current_energy=self.energy_level,
            min_energy=self.energy_min,
            max_energy=self.energy_max,
            charge_rate_max=self.charge_rate_max,
            discharge_rate_max=self.discharge_rate_max
        )
        raw_energy_change = self.energy_level - previous_energy
        self.energy_change = raw_energy_change

        # Final logging
        self.logger.info(f"Battery energy level changed from {previous_energy:.3f} to {self.energy_level:.3f} MWh")

    def get_state(self) -> float:
        """
        Retrieves the current energy level of the battery.

        Returns:
            float: Current energy level in MWh.
        """
        self.logger.debug(f"Retrieving battery state: {self.energy_level} MWh")
        return self.energy_level

    def update(self, time: float, action: float = 0.0) -> None:
        """
        Updates the battery's state based on dynamics, time, and action.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).
            action (float, optional): Action to perform (default is 0.0).
                                       Positive for charging, negative for discharging.
        """
        self.logger.debug(f"Updating Battery at time: {time} with action: {action} MW")
        self.current_time = time
        self.perform_action(action)

    def reset(self, initial_level: Optional[float] = None) -> None:
        """
        Resets the battery to specified or default initial level.
        
        Args:
            initial_level: Optional override for initial energy level
        """
        if initial_level is not None:
            self.energy_level = initial_level
            self.logger.info(f"Reset Battery to specified level: {self.energy_level} MWh")
        else:
            self.energy_level = self.initial_energy
            self.logger.info(f"Reset Battery to default level: {self.energy_level} MWh")
        self.logger.debug(f"Battery reset complete. Current energy level: {self.energy_level} MWh")

