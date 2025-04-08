import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, DefaultDict
from collections import defaultdict
import statistics

# Import from datamodel
from datamodel import OrderDepth, Order, Trade, TradingState, Symbol

class MarketAnalyzer:
    """
    Analyze market data to understand patterns and behavior of different products.
    """
    def __init__(self, data_path: str):
        """
        Initialize with path to historical market data.
        """
        self.data_path = data_path
        self.trading_data = self.load_trading_data()
        self.time_steps = len(self.trading_data)
        self.products = self.identify_products()
        print(f"Loaded {self.time_steps} time steps with {len(self.products)} products: {self.products}")
        
    def load_trading_data(self):
        """Load historical market data from file."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading trading data: {e}")
            return []
    
    def identify_products(self):
        """Identify available products in the data."""
        products = set()
        for state_data in self.trading_data:
            if "order_depths" in state_data:
                for product in state_data["order_depths"].keys():
                    products.add(product)
        return list(products)
    
    def extract_price_series(self):
        """Extract price series for each product."""
        mid_prices = {product: [] for product in self.products}
        timestamps = []
        
        for state_data in self.trading_data:
            timestamps.append(state_data.get("timestamp", 0))
            
            for product in self.products:
                # Extract order depth
                if "order_depths" in state_data and product in state_data["order_depths"]:
                    depth = state_data["order_depths"][product]
                    buy_orders = depth.get("buy_orders", {})
                    sell_orders = depth.get("sell_orders", {})
                    
                    # Calculate mid price
                    best_bid = max(buy_orders.keys()) if buy_orders else None
                    best_ask = min(sell_orders.keys()) if sell_orders else None
                    
                    if best_bid is not None and best_ask is not None:
                        mid_price = (best_bid + best_ask) / 2
                    else:
                        mid_price = None
                    
                    mid_prices[product].append(mid_price)
                else:
                    mid_prices[product].append(None)
        
        # Create DataFrame
        df = pd.DataFrame(mid_prices)
        df["timestamp"] = timestamps
        df.set_index("timestamp", inplace=True)
        
        # Forward fill missing values
        df = df.ffill()
        
        return df
    
    def extract_order_book_metrics(self):
        """Extract order book metrics like spread, depth, and imbalance."""
        metrics = {product: {
            "spread": [],
            "bid_depth": [],
            "ask_depth": [],
            "imbalance": []
        } for product in self.products}
        
        timestamps = []
        
        for state_data in self.trading_data:
            timestamps.append(state_data.get("timestamp", 0))
            
            for product in self.products:
                # Extract order depth
                if "order_depths" in state_data and product in state_data["order_depths"]:
                    depth = state_data["order_depths"][product]
                    buy_orders = depth.get("buy_orders", {})
                    sell_orders = depth.get("sell_orders", {})
                    
                    # Calculate spread
                    best_bid = max(buy_orders.keys()) if buy_orders else None
                    best_ask = min(sell_orders.keys()) if sell_orders else None
                    
                    if best_bid is not None and best_ask is not None:
                        spread = best_ask - best_bid
                    else:
                        spread = None
                    
                    # Calculate depth
                    bid_depth = sum(buy_orders.values()) if buy_orders else 0
                    ask_depth = sum(abs(v) for v in sell_orders.values()) if sell_orders else 0
                    
                    # Calculate imbalance
                    if bid_depth + ask_depth > 0:
                        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
                    else:
                        imbalance = 0
                    
                    metrics[product]["spread"].append(spread)
                    metrics[product]["bid_depth"].append(bid_depth)
                    metrics[product]["ask_depth"].append(ask_depth)
                    metrics[product]["imbalance"].append(imbalance)
                else:
                    metrics[product]["spread"].append(None)
                    metrics[product]["bid_depth"].append(0)
                    metrics[product]["ask_depth"].append(0)
                    metrics[product]["imbalance"].append(0)
        
        # Create DataFrames
        dfs = {}
        for product in self.products:
            df = pd.DataFrame(metrics[product])
            df["timestamp"] = timestamps
            df.set_index("timestamp", inplace=True)
            df = df.ffill()  # Forward fill missing values
            dfs[product] = df
        
        return dfs
    
    def analyze_trade_patterns(self):
        """Analyze trade patterns to identify market behavior."""
        trade_data = {product: {
            "prices": [],
            "volumes": [],
            "buy_sell_ratio": [],
            "timestamps": []
        } for product in self.products}
        
        for state_data in self.trading_data:
            timestamp = state_data.get("timestamp", 0)
            
            for product in self.products:
                # Extract trades
                if "trades" in state_data and product in state_data["trades"]:
                    trades = state_data["trades"][product]
                    if trades:
                        # Extract prices and volumes
                        prices = [t.get("price", 0) for t in trades]
                        volumes = [t.get("quantity", 0) for t in trades]
                        
                        # Calculate buy/sell ratio
                        buy_volume = sum(v for v in volumes if v > 0)
                        sell_volume = sum(abs(v) for v in volumes if v < 0)
                        
                        if buy_volume + sell_volume > 0:
                            buy_sell_ratio = buy_volume / (buy_volume + sell_volume)
                        else:
                            buy_sell_ratio = 0.5
                        
                        # Append to data
                        trade_data[product]["prices"].append(statistics.mean(prices))
                        trade_data[product]["volumes"].append(sum(abs(v) for v in volumes))
                        trade_data[product]["buy_sell_ratio"].append(buy_sell_ratio)
                        trade_data[product]["timestamps"].append(timestamp)
        
        # Create DataFrames
        dfs = {}
        for product in self.products:
            if trade_data[product]["timestamps"]:
                df = pd.DataFrame({
                    "price": trade_data[product]["prices"],
                    "volume": trade_data[product]["volumes"],
                    "buy_sell_ratio": trade_data[product]["buy_sell_ratio"],
                })
                df["timestamp"] = trade_data[product]["timestamps"]
                df.set_index("timestamp", inplace=True)
                dfs[product] = df
            else:
                dfs[product] = pd.DataFrame()
        
        return dfs
    
    def calculate_volatility(self, window=10):
        """Calculate rolling volatility of price series."""
        price_df = self.extract_price_series()
        volatility = {}
        
        for product in self.products:
            if product in price_df.columns:
                # Calculate returns
                returns = price_df[product].pct_change().dropna()
                
                # Calculate rolling standard deviation
                vol = returns.rolling(window=window).std() * np.sqrt(window)
                volatility[product] = vol
        
        # Create DataFrame
        df = pd.DataFrame(volatility)
        
        return df
    
    def calculate_correlations(self):
        """Calculate correlations between products."""
        price_df = self.extract_price_series()
        
        # Calculate returns
        returns = price_df.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        return corr_matrix
    
    def identify_market_regimes(self, n_clusters=3):
        """Identify market regimes using clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Extract features
            price_df = self.extract_price_series()
            returns = price_df.pct_change().dropna()
            
            # Combine with order book metrics
            features = []
            
            for product in self.products:
                metrics_df = self.extract_order_book_metrics()[product]
                
                # Resample to match