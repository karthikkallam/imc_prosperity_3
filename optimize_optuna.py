import optuna
import jsonpickle
import numpy as np
import pandas as pd
import os
import pickle
import json
from collections import defaultdict
import math
from typing import Dict, List, Any, Tuple, Optional, DefaultDict

# Import simulation classes
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from trader import Trader, Product, PARAMS

# Import evaluation metrics
from metrics import calculate_pnl, calculate_position_value, calculate_sharpe_ratio

class BacktestEnvironment:
    """Simple backtesting environment to simulate trading over historical data."""
    def __init__(self, data_path: str):
        """
        Initialize with path to historical market data.
        """
        self.data_path = data_path
        self.trading_data = self.load_trading_data()
        self.time_steps = len(self.trading_data)
        print(f"Loaded {self.time_steps} time steps of trading data")
        
    def load_trading_data(self):
        """
        Load historical market data for backtesting.
        Expected format: List of state dictionaries compatible with TradingState.
        """
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading trading data: {e}")
            # Return empty data as fallback
            return []
    
    def simulate_trading(self, trader: Trader):
        """
        Run trading simulation using the provided trader instance.
        
        Returns:
            tuple: (pnl, metrics)
        """
        # Initialize tracking variables
        positions: Dict[str, List[int]] = defaultdict(list)
        cash_flows: Dict[str, List[float]] = defaultdict(list)
        executed_trades: Dict[str, List[Trade]] = defaultdict(list)
        market_values: Dict[str, List[float]] = defaultdict(list)
        
        # Track trader data over time
        trader_data = ""
        
        # Process each time step
        for step, state_data in enumerate(self.trading_data):
            # Convert state_data to TradingState
            state = self.dict_to_trading_state(state_data, trader_data)
            
            # Execute trader logic
            orders, conversions, trader_data = trader.run(state)
            
            # Track positions and cash flows
            for product in state.order_depths.keys():
                position = state.position.get(product, 0)
                positions[product].append(position)
                
                # Calculate market value of position
                mid_price = self.get_mid_price(state.order_depths.get(product, OrderDepth()))
                if mid_price:
                    market_value = position * mid_price
                    market_values[product].append(market_value)
                
                # Process executed orders for this step
                if product in orders:
                    cash_flow = 0
                    for order in orders[product]:
                        # Simulate execution at order price
                        price = order.price
                        quantity = order.quantity
                        cash_flow -= price * quantity  # Negative cash flow for buys, positive for sells
                        
                        # Record trade
                        executed_trades[product].append(
                            Trade(product, price, quantity, 'B' if quantity > 0 else 'S', "BACKTEST")
                        )
                        
                    cash_flows[product].append(cash_flow)
        
        # Calculate PnL and other metrics
        metrics = self.calculate_metrics(positions, cash_flows, market_values, executed_trades)
        return metrics["total_pnl"], metrics
    
    def dict_to_trading_state(self, state_dict: Dict, trader_data: str) -> TradingState:
        """Convert dictionary representation to TradingState."""
        # Create empty order depths
        order_depths = {}
        for product, depth in state_dict.get("order_depths", {}).items():
            od = OrderDepth()
            od.buy_orders = depth.get("buy_orders", {})
            od.sell_orders = depth.get("sell_orders", {})
            order_depths[product] = od
        
        # Convert trades
        trades = {}
        for product, trade_list in state_dict.get("trades", {}).items():
            trades[product] = [
                Trade(t.get("symbol"), t.get("price"), 
                      t.get("quantity"), t.get("buyer"), t.get("seller"))
                for t in trade_list
            ]
        
        # Create TradingState
        return TradingState(
            timestamp=state_dict.get("timestamp", 0),
            listings=state_dict.get("listings", {}),
            order_depths=order_depths,
            trades=trades,
            own_trades=state_dict.get("own_trades", {}),
            position=state_dict.get("position", {}),
            observations=state_dict.get("observations", {}),
            traderData=trader_data
        )
    
    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate mid price from order depth."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def calculate_metrics(self, positions, cash_flows, market_values, executed_trades):
        """Calculate performance metrics."""
        metrics = {}
        
        # Calculate PnL for each product
        pnl_by_product = {}
        for product in positions.keys():
            # Final position value minus cash spent/received
            final_position = positions[product][-1] if positions[product] else 0
            final_market_value = market_values[product][-1] if market_values[product] else 0
            total_cash_flow = sum(cash_flows[product])
            
            product_pnl = final_market_value + total_cash_flow
            pnl_by_product[product] = product_pnl
        
        metrics["pnl_by_product"] = pnl_by_product
        metrics["total_pnl"] = sum(pnl_by_product.values())
        
        # Calculate Sharpe ratio if we have enough data
        daily_returns = []
        for product in market_values.keys():
            if len(market_values[product]) > 1:
                returns = np.diff(market_values[product]) / market_values[product][:-1]
                daily_returns.extend(returns)
        
        if daily_returns:
            metrics["sharpe_ratio"] = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
        else:
            metrics["sharpe_ratio"] = 0
            
        # Count trades
        metrics["trade_count"] = {
            product: len(trades) for product, trades in executed_trades.items()
        }
        
        return metrics

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        float: Objective value (PnL to maximize)
    """
    # Define hyperparameter search space
    params = {
        Product.RAINFOREST_RESIN: {
            "anchor_value": trial.suggest_int("rr_anchor", 9800, 10200),
            "use_ema": trial.suggest_categorical("rr_use_ema", [True, False]),
            "ema_alpha": trial.suggest_float("rr_ema_alpha", 0.05, 0.5),
            "take_width": trial.suggest_int("rr_take_width", 1, 5),
            "clear_width": trial.suggest_int("rr_clear_width", 0, 3),
            "disregard_edge": trial.suggest_int("rr_disregard_edge", 1, 3),
            "join_edge": trial.suggest_int("rr_join_edge", 0, 3),
            "default_edge": trial.suggest_int("rr_default_edge", 1, 5),
            "skew_factor": trial.suggest_float("rr_skew_factor", 0.01, 0.1),
            "soft_limit_ratio": trial.suggest_float("rr_soft_limit_ratio", 0.5, 0.9)
        },
        Product.KELP: {
            "use_ema": trial.suggest_categorical("kelp_use_ema", [True, False]),
            "ema_alpha": trial.suggest_float("kelp_ema_alpha", 0.1, 0.6),
            "reversion_beta": trial.suggest_float("kelp_reversion_beta", -0.5, 0.0),
            "take_width": trial.suggest_int("kelp_take_width", 1, 5),
            "clear_width": trial.suggest_int("kelp_clear_width", 0, 3),
            "prevent_adverse": trial.suggest_categorical("kelp_prevent_adverse", [True, False]),
            "adverse_volume": trial.suggest_int("kelp_adverse_volume", 5, 30),
            "disregard_edge": trial.suggest_int("kelp_disregard_edge", 1, 3),
            "join_edge": trial.suggest_int("kelp_join_edge", 0, 3),
            "default_edge": trial.suggest_int("kelp_default_edge", 1, 4),
            "skew_factor": trial.suggest_float("kelp_skew_factor", 0.01, 0.15),
            "soft_limit_ratio": trial.suggest_float("kelp_soft_limit_ratio", 0.5, 0.9)
        }
    }
    
    # Create trader with trial parameters
    trader = Trader(params)
    
    # Run backtesting
    env = BacktestEnvironment("round1_data.json")
    pnl, metrics = env.simulate_trading(trader)
    
    # Log metrics for this trial
    print(f"Trial {trial.number} - PnL: {pnl:.2f}, Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    return pnl  # We want to maximize PnL

def save_best_params(study, filename="best_params.json"):
    """Save the best parameters to a file."""
    best_params = study.best_params
    
    # Convert to parameter format expected by Trader
    formatted_params = {
        Product.RAINFOREST_RESIN: {},
        Product.KELP: {}
    }
    
    # Map parameter names to their respective products
    for param_name, value in best_params.items():
        if param_name.startswith("rr_"):
            key = param_name[3:]  # Remove "rr_" prefix
            formatted_params[Product.RAINFOREST_RESIN][key] = value
        elif param_name.startswith("kelp_"):
            key = param_name[5:]  # Remove "kelp_" prefix
            formatted_params[Product.KELP][key] = value
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(formatted_params, f, indent=2)
    
    print(f"Best parameters saved to {filename}")
    return formatted_params

if __name__ == "__main__":
    print("Starting hyperparameter optimization...")
    
    # Create study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    print(f"\nBest PnL: {study.best_value:.2f}")
    
    # Save best parameters
    best_params = save_best_params(study)
    
    # Create and evaluate trader with best parameters
    best_trader = Trader(best_params)
    env = BacktestEnvironment("round1_data.json")
    final_pnl, metrics = env.simulate_trading(best_trader)
    
    print("\nFinal Evaluation:")
    print(f"Total PnL: {final_pnl:.2f}")
    print(f"PnL by Product: {metrics['pnl_by_product']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Trade Count: {metrics['trade_count']}")