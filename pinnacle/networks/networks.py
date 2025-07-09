# networks/networks.py
"""
Neural network architectures and management for PINNACLE.
"""
import torch
import torch.nn as nn
import torch.onnx
from typing import Dict, Any, Optional, List


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class Swoosh(nn.Module):
    """Swoosh activation function: |x| * sigmoid(x)"""

    def forward(self, x):
        return torch.abs(x) * torch.sigmoid(x)


class Swash(nn.Module):
    """Swash activation function: x² * sigmoid(x)"""

    def forward(self, x):
        return x ** 2 * torch.sigmoid(x)


class SquashSwish(nn.Module):
    """SquashSwish activation function: x * sigmoid(x) + 0.5"""

    def forward(self, x):
        return x * torch.sigmoid(x) + 0.5


class FFN(nn.Module):
    """
    Fully Connected Feed Forward Neural Network.

    Args:
        input_dim: Number of input features
        output_dim: Number of output features  
        hidden_layers: Number of hidden layers
        layer_size: Size of each hidden layer
        activation: Activation function name ('swish', 'swoosh', 'swash', 'squash_swish', 'relu', 'tanh')
        initialize_weights: Whether to apply Xavier initialization
    """

    def __init__(
            self,
            input_dim: int = 2,
            output_dim: int = 1,
            hidden_layers: int = 5,
            layer_size: int = 20,
            activation: str = "swish",
            initialize_weights: bool = False
    ):
        super(FFN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = hidden_layers
        self.layer_size = layer_size

        # Select activation function
        activation_map = {
            "swish": Swish(),
            "swoosh": Swoosh(),
            "swash": Swash(),
            "squash_swish": SquashSwish(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh()
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_map.keys())}")

        self.activation = activation_map[activation]

        # Input layer
        self.input_layer = nn.Linear(input_dim, self.layer_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.layer_size, self.layer_size)
            for _ in range(self.num_layers)  # Fixed: was "for * in range"
        ])

        # Output layer
        self.output_layer = nn.Linear(self.layer_size, output_dim)

    def forward(self, x):
        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers: 
            x = self.activation(layer(x))

        return self.output_layer(x)


class NetworkManager:
    """
    Manages multiple neural networks for the PINNACLE system.

    Handles creation, initialization, and management of all networks
    needed for the electrochemical PINN.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize all networks based on configuration.

        Args:
            config: Configuration dictionary with network specifications
            device: PyTorch device to place networks on
        """
        self.config = config
        self.device = device
        self.networks = {}

        # Create all required networks
        self._create_networks()

    def _create_networks(self):
        """Create all neural networks based on configuration"""
        arch_config = self.config['arch']

        # Network specifications: (name, input_dim, config_key)
        network_specs = [
            ('potential', 3, 'potential'),  # (x, t, E) -> φ_hat
            ('cv', 3, 'CV'),  # (x, t, E) -> c_hat_cv
            ('av', 3, 'AV'),  # (x, t, E) -> c_hat_av
            ('h', 3, 'h'),  # (x, t, E) -> c_hat_h
            ('film_thickness', 2, 'L'),  # (t, E) -> L_hat
        ]

        for net_name, input_dim, config_key in network_specs:
            self.networks[net_name] = FFN(
                input_dim=input_dim,
                output_dim=1,
                hidden_layers=arch_config[config_key]['hidden_layers'],
                layer_size=arch_config[config_key]['layer_size'],
                activation="swish",  # Can make this configurable too
                initialize_weights=False
            ).to(self.device)

    def get_network(self, name: str) -> nn.Module:
        """Get a specific network by name"""
        if name not in self.networks:
            raise KeyError(f"Network '{name}' not found. Available: {list(self.networks.keys())}")
        return self.networks[name]
    
    def get_all_parameters_ordered(self) -> List[torch.nn.Parameter]:
            """Get parameters in same order as monolith"""
            # Match monolith order: potential, CV, AV, h, L
            ordered_names = ['potential', 'cv', 'av', 'h', 'film_thickness']
            params = []
            for name in ordered_names:
                if name in self.networks:
                    params.extend(list(self.networks[name].parameters()))
            return params

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for all networks"""
        return {name: network.state_dict() for name, network in self.networks.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict for all networks"""
        for name, network_state in state_dict.items():
            if name in self.networks:
                self.networks[name].load_state_dict(network_state)
            else:
                print(f"Warning: Network '{name}' not found in current model")

    def train(self):
        """Set all networks to training mode"""
        for network in self.networks.values():
            network.train()

    def eval(self):
        """Set all networks to evaluation mode"""
        for network in self.networks.values():
            network.eval()

    def to(self, device: torch.device):
        """Move all networks to specified device"""
        for network in self.networks.values():
            network.to(device)
        self.device = device

    def export_to_onnx(self, save_path: str = "model.onnx"):
        """
        Export all networks to ONNX format for visualization/deployment.

        Args:
            save_path: Path to save ONNX file
        """

        # Create a combined model for export
        class CombinedModel(nn.Module):
            def __init__(self, networks):
                super().__init__()
                self.potential_network = networks['potential']
                self.cv_network = networks['cv']
                self.av_network = networks['av']
                self.h_network = networks['h']
                self.film_thickness_network = networks['film_thickness']

            def forward(self, x, t, E):
                # Spatial-temporal-potential inputs
                xte_input = torch.cat([x, t, E], dim=1)
                # Temporal-potential inputs
                te_input = torch.cat([t, E], dim=1)

                potential = self.potential_network(xte_input)
                cv_conc = self.cv_network(xte_input)
                av_conc = self.av_network(xte_input)
                h_conc = self.h_network(xte_input)
                thickness = self.film_thickness_network(te_input)

                return potential, cv_conc, av_conc, h_conc, thickness

        # Move to CPU for export
        cpu_networks = {}
        for name, net in self.networks.items():
            cpu_networks[name] = net.cpu().eval()

        combined = CombinedModel(cpu_networks)

        # Export to ONNX
        dummy_x = torch.randn(1, 1)  # spatial position
        dummy_t = torch.randn(1, 1)  # time
        dummy_E = torch.randn(1, 1)  # applied potential

        torch.onnx.export(
            combined,
            (dummy_x, dummy_t, dummy_E),
            save_path,
            input_names=['x_position', 't_time', 'E_applied'],
            output_names=['potential', 'cv_concentration', 'av_concentration',
                          'hole_concentration', 'film_thickness'],
            dynamic_axes={
                'x_position': {0: 'batch_size'},
                't_time': {0: 'batch_size'},
                'E_applied': {0: 'batch_size'}
            }
        )

        # Move back to original device
        for network in self.networks.values():
            network.to(self.device).train()

        print(f"✅ Networks exported to {save_path}")

    def __getitem__(self, key: str) -> nn.Module:
        """Allow dict-like access to networks"""
        return self.get_network(key)

    def __contains__(self, key: str) -> bool:
        """Check if network exists"""
        return key in self.networks

    def keys(self):
        """Get network names"""
        return self.networks.keys()


