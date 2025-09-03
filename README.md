# MPAC-Energy-Net Integration: Multi-Perspective Actor Critic for Power Grid Management

This repository contains an innovative integration of the **Multi-Perspective Actor Critic (MPAC)** algorithm with **energy-net**, a sophisticated power grid simulation framework. The project focuses on applying robust and safe deep reinforcement learning to power consumption and storage (PCS) unit management in electricity markets.

## Project Overview

The integration enables intelligent battery management systems that can:
- **Optimize battery charging/discharging** based on real-time electricity prices
- **Adapt to market dynamics** using trained ISO (Independent System Operator) models
- **Handle uncertainty and safety constraints** through MPAC's robust learning approach
- **Manage energy storage** in response to demand patterns and cost structures

##  Architecture

### Core Components

#### 1. **MPAC Algorithm Integration**
- **Multi-Perspective Actor Critic**: Implements risk-aware value decomposition for robust RL
- **Safety Constraints**: Built-in support for safe reinforcement learning with CRPO and Lagrange methods
- **Robustness Methods**: RAMU (Risk-Aware Model Uncertainty) for handling model uncertainties
- **Multi-Objective Learning**: Support for balancing multiple objectives simultaneously

#### 2. **Energy-Net Environment**
- **PCS Unit Environment**: Simulates power consumption and storage units
- **ISO Market Simulation**: Models electricity market dynamics and pricing
- **Battery Dynamics**: Realistic battery charging/discharging with efficiency models
- **Demand Patterns**: Configurable demand variations (sinusoidal, double-peak, random)

#### 3. **Integration Layer**
- **MPAC Wrapper**: Stable-Baselines3 compatible interface for MPAC
- **Environment Wrappers**: Seamless integration between MPAC and energy-net
- **Configuration Management**: Centralized settings for environment parameters

##  Key Features

### **Battery Management**
- **Dynamic Charging/Discharging**: Real-time control based on market prices
- **Efficiency Modeling**: Configurable charge/discharge efficiency parameters
- **Capacity Constraints**: Configurable battery capacity and rate limits
- **Perturbation Support**: Built-in robustness testing with parameter variations

### **Market Integration**
- **ISO Policy Integration**: Pre-trained ISO models for price determination
- **Quadratic Pricing**: Support for realistic electricity pricing models
- **Demand Response**: Adaptive strategies based on demand patterns
- **Cost Optimization**: Minimize energy costs while maintaining grid stability

### **Robust Learning**
- **Uncertainty Handling**: MPAC's multi-perspective approach for model uncertainty
- **Safety Constraints**: Built-in safety mechanisms for grid operations
- **Domain Randomization**: Support for training robustness across different scenarios
- **Multi-Objective Optimization**: Balance cost, efficiency, and safety objectives

## Safety Constraint System

The project implements a comprehensive safety constraint system specifically designed for energy grid operations, ensuring safe battery management and thermal control.

### **Available Safety Constraints**

#### **1. Battery Level Constraint (`battery`)**
- **Purpose**: Prevents battery from operating outside safe charge levels
- **Bounds**: 
  - Lower bound: 20% (prevents over-discharge)
  - Upper bound: 80% (prevents over-charge)
- **Cost Function**: Returns cost in [0,1] range (0 = safe, 1 = violation)
- **Activation**: Automatically penalizes when battery level approaches unsafe ranges

#### **2. Thermal Constraint (`thermal`)**
- **Purpose**: Monitors battery temperature to prevent thermal runaway
- **Bounds**: 
  - Lower bound: 15°C (prevents cold damage)
  - Upper bound: 35°C (prevents overheating)
- **Thermal Model**: Implements Joule heating + linear cooling physics
- **Cost Function**: Temperature-dependent violation cost with exponential scaling

### **How to Activate Safety Constraints**

#### **Command Line Activation**
```bash
# Enable safety constraints with default settings
python -m MPAC.train --env_type energy_net --env_name pcs --task_name pcs_unit --safe

# Enable specific constraints
python -m MPAC.train --env_type energy_net --env_name pcs --task_name pcs_unit --safe --energy_constraints battery thermal

# Enable all available constraints
python -m MPAC.train --env_type energy_net --env_name pcs --task_name pcs_unit --safe --energy_constraints_all

# Adjust safety coefficient (0.0 = impossible, 1.0 = baseline, higher = stricter)
python -m MPAC.train --env_type energy_net --env_name pcs --task_name pcs_unit --safe --safety_coeff 0.5
```

#### **Configuration File Settings**
```yaml
# In your environment configuration
energy_constraints: ["battery", "thermal"]  # Enable specific constraints
energy_constraints_all: true               # Enable all constraints
safety_coeff: 1.0                         # Safety constraint strength
```

### **Safety Constraint Implementation Details**

#### **Constraint Function Interface**
```python
def constraint_function(obs, info, previous_state=None):
    """
    Args:
        obs: Current observation
        info: Environment info dictionary
        previous_state: Previous state for temporal constraints
    
    Returns:
        float: Cost in [0,1] range (0 = safe, 1 = violation)
    """
    # Constraint logic here
    return cost_value
```

#### **Cost Calculation**
- **Battery Level**: Normalized distance from safe bounds
- **Thermal**: Temperature deviation from safe range with exponential scaling
- **Temporal Penalties**: Additional costs for static battery levels (no change)

#### **Safety Coefficient Scaling**
- **safety_coeff = 0.0**: Constraints are impossible to satisfy
- **safety_coeff = 0.5**: Constraints are moderately strict
- **safety_coeff = 1.0**: Standard constraint difficulty
- **safety_coeff > 1.0**: Constraints become stricter and harder to satisfy

### **Integration with MPAC Algorithm**

The safety constraints are seamlessly integrated with MPAC's robust learning framework:

1. **Constraint Monitoring**: Real-time evaluation of safety violations
2. **Cost Integration**: Safety costs are incorporated into the learning objective
3. **Policy Adaptation**: MPAC automatically learns to balance performance with safety
4. **Robustness**: Constraints work with RAMU and other robust methods

### **Safety Constraint Benefits**

- **Grid Stability**: Prevents battery operations that could destabilize the grid
- **Equipment Protection**: Extends battery lifespan through safe operating ranges
- **Regulatory Compliance**: Meets safety standards for energy storage systems
- **Risk Management**: Quantifies and manages operational risks## 📋 Requirements

### **System Requirements**
- Python >= 3.8, <= 3.10
- CUDA >= 11.1 (for GPU acceleration)
- Sufficient RAM for large-scale simulations

### **Dependencies**
The project uses a comprehensive conda environment with the following key packages:

```yaml
# Core ML/RL
- pytorch
- stable-baselines3
- gymnasium
- numpy
- scipy

# Energy Simulation
- dm-control
- realworldrl_suite

# Visualization & Analysis
- matplotlib
- seaborn
- jupyter

# Utilities
- absl-py
- tqdm
- h5py
```

### **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd MPAC_STAV

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate MPAC

# Install energy-net dependencies
pip install -e energy_net/
```

## 🎮 Usage

### **Training PCS Agents**

#### **Basic Training**
```bash
python -m MPAC.train_wrap --env_type energy_net --env_name pcs --task_name pcs_unit --total_timesteps 1000000
```

#### **Safe RL Training**
```bash
python -m MPAC.train_wrap --env_type energy_net --env_name pcs --task_name pcs_unit --safe 
```

#### **Robust Training with RAMU**
```bash
python -m MPAC.train_wrap --env_type energy_net --env_name pcs --task_name pcs_unit --robust --robust_type ramu
```

#### **Multi-Objective Training**
```bash
python -m MPAC.train_wrap --env_type energy_net --env_name pcs --task_name pcs_unit --multiobj_enable --multiobj_coeff 0.5
```

### *MPAC training (both robust and safe)
```bash
python -m MPAC.train_wrap --env_type energy_net --env_name pcs --task_name pcs_unit --robust --robust_type ramu --safe
```

### **Evaluation**
```bash
python -m MPAC.eval --import_file <train_filename> --import_path logs/
```
### **Safety_constraints**
```bash
Add --energy_constraint battery/thermal or --energy_constraints_all to the command line of train and eval for safety constraints addition.
```
### **Visualization**

The project provides multiple plotting utilities for comprehensive analysis:

#### **Basic Plotting**
```bash
# Plot training metrics
python -m MPAC.plot --plot_type train --metrics J_tot Jc_tot --import_files logs/<train_filename>

# Plot evaluation results
python -m MPAC.plot --plot_type eval --metrics J_tot Jc_tot --import_files logs/<eval_filename>
```

#### **Advanced Analysis Tools**

**1. Plot Chart (`plot_chart.py`)**
- **Rank Distribution Analysis**: Compares algorithm performance across perturbation scenarios
- **Stacked Bar Charts**: Visualizes how algorithms rank against each other
- **Performance Benchmarking**: MPO, CRPO, RAMU, and MPAC comparison

**2. Plot Table (`plot_table.py`)**
- **Normalized Performance Tables**: Compares algorithms relative to MPO baseline
- **CSV and LaTeX Output**: Generates publication-ready tables
- **Statistical Summaries**: Mean performance across perturbation ranges

**3. Plot Team (`plot_team.py`)**
- **Multi-Algorithm Comparison**: Side-by-side reward and cost analysis
- **Perturbation Response**: Shows how algorithms respond to parameter changes
- **Standard Deviation Visualization**: Includes uncertainty bands for robust comparison

**Usage Examples:**
```bash
# Generate rank distribution charts
python MPAC/plot_chart.py

# Create performance summary tables
python MPAC/plot_table.py

# Generate team comparison plots
python MPAC/plot_team.py
```

## Configuration

### **Environment Configuration**

The project uses YAML configuration files for easy customization:

#### **PCS Unit Configuration** (`configs/pcs_unit_config.yaml`)
```yaml
battery:
  max: 100.0          # Maximum battery capacity (MWh)
  charge_rate_max: 10.0  # Maximum charging rate (MW)
  discharge_rate_max: 10.0  # Maximum discharging rate (MW)
  charge_efficiency: 1.0   # Charging efficiency
  discharge_efficiency: 1.0 # Discharging efficiency

action:
  multi_action: false  # Single action (battery control only)
  consumption_action:
    enabled: false     # Disable consumption control
  production_action:
    enabled: false     # Disable production control
```

#### **ISO Configuration** (`configs/iso_config.yaml`)
```yaml
pricing_policy: ONLINE
cost_type: CONSTANT
demand_pattern: DOUBLE_PEAK
```

### **MPAC Configuration**

Key MPAC parameters can be configured through command-line arguments:

- `--safe`: Enable safety constraints
- `--robust`: Enable robustness methods
- `--multiobj_enable`: Enable multi-objective learning
- `--total_timesteps`: Training duration
- `--checkpoint_file`: Model checkpointing

## Research Applications

### **Grid Stability**
- **Frequency Regulation**: Maintain grid frequency through battery response
- **Peak Shaving**: Reduce peak demand through intelligent storage
- **Renewable Integration**: Smooth renewable energy fluctuations

### **Market Optimization**
- **Arbitrage**: Buy low, sell high based on price predictions
- **Demand Response**: Adapt consumption to market conditions
- **Capacity Markets**: Participate in capacity and ancillary services

### **Safety and Robustness**
- **Grid Constraints**: Respect operational limits and safety margins
- **Uncertainty Handling**: Adapt to changing market conditions
- **Fault Tolerance**: Maintain operation under various failure scenarios

##  Results and Analysis

### **Training Metrics**
- **Total Reward (J_tot)**: Overall performance across objectives
- **Total Cost (Jc_tot)**: Safety constraint violations
- **Battery Efficiency**: Energy storage utilization
- **Market Performance**: Cost savings and revenue generation

### **Evaluation Scenarios**
- **Nominal Conditions**: Standard operating conditions
- **Perturbed Environments**: Robustness testing with parameter variations
- **Safety Violations**: Testing constraint handling
- **Market Shocks**: Response to sudden price changes

##  Development

### **Project Structure**
```
MPAC_STAV/
├── MPAC/                    # MPAC algorithm implementation
│   ├── algs/               # Algorithm implementations
│   ├── actors/             # Actor networks
│   ├── critics/            # Critic networks
│   ├── envs/               # Environment wrappers
│   ├── robust_methods/     # Robustness implementations
│   └── train_wrap.py       # Training wrapper
├── energy_net/             # Energy simulation framework
│   ├── env/                # Environment implementations
│   ├── controllers/        # PCS and ISO controllers
│   ├── dynamics/           # Physical dynamics models
│   └── market/             # Market simulation
├── configs/                # Configuration files
└── environment.yml         # Conda environment
```

### **Adding New Features**
1. **New Battery Models**: Extend `energy_net/dynamics/storage_dynamics/`
2. **Custom Rewards**: Implement in `energy_net/rewards/`
3. **Additional Controllers**: Add to `energy_net/controllers/`
4. **MPAC Extensions**: Modify `MPAC/algs/` and `MPAC/robust_methods/`

## References

- **MPAC Algorithm**: Multi-Perspective Actor Critic for robust RL
- **Energy-Net**: Power grid simulation framework
- **Safe RL**: Constrained reinforcement learning methods
- **Battery Management**: Energy storage optimization strategies


**Note**: This project represents a significant integration effort between advanced reinforcement learning algorithms and power grid simulation, enabling intelligent energy management systems that can adapt to real-world uncertainties while maintaining safety and efficiency.

