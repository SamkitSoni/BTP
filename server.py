# server.py (fixed for new Flower versions)
import flwr as fl
from flwr.server.strategy import FedAvg
import argparse

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,  # Pass round number to clients
        "round": server_round,  # Alternative key for compatibility
    }
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=fit_config,  # Pass config function to send round numbers
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
