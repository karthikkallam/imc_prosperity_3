import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datamodel import Trade, OrderDepth

def calculate_pnl(positions: Dict[str, List[int]], 
                 cash_flows: Dict[str, List[float]], 
                 final_prices: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate PnL for each product.
    
    Args:
        positions: Dict mapping product to list of positions over time
        cash_flows: Dict mapping product to list of cash flows over time
        final_prices: Dict mapping product to final price
        
    Returns:
        Dict mapping product to final PnL
    """
    pnl = {}
    for product in positions.keys():
        if not positions[product]:  # Skip if no positions
            pnl[product] = 0
            continue
            
        # Final position value
        final_pos = positions[product][-1]
        final_value = final_pos * final_prices.get(product, 0)
        
        # Sum of all cash flows (negative for buys, positive for sells)
        total_cash_flow = sum(cash_flows.get(product, [0]))
        
        # PnL = final position value + cash flows
        pnl[product] = final_value + total_cash_flow
    
    return pnl

def calculate_position_value(positions: Dict[str, int], 
                           prices: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate mark-to-market value of positions.
    
    Args:
        positions: Dict mapping product to current position
        prices: Dict mapping product to current price
        
    Returns:
        Dict mapping product to position value
    """
    values = {}
    for product, position in positions.items():
        price = prices.get(product, 0)
        values[product] = position * price
    
    return values

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from a list of returns.
    
    Args:
        returns: List of return values
        risk_free_rate: Risk-free rate (default 0)
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Calculate Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)  # Using sample standard deviation
    
    if std_dev == 0:
        return 0.0  # Avoid division by zero
    
    sharpe = mean_excess_return / std_dev
    
    # Annualize if necessary (example for daily returns)
    # sharpe_annualized = sharpe * np.sqrt(252)  # 252 trading days in a year
    
    return sharpe

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from an equity curve.
    
    Args:
        equity_curve: List of equity values over time
        
    Returns:
        Maximum drawdown as a percentage
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percentage terms
    drawdown = (running_max - equity_curve) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.max(drawdown)
    
    return max_drawdown

def calculate_win_rate(trades: List[Trade]) -> float:
    """
    Calculate win rate from a list of trades.
    
    Args:
        trades: List of Trade objects
        
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
    
    profitable_trades = sum(1 for trade in trades if trade.price * trade.quantity > 0)
    win_rate = profitable_trades / len(trades)
    
    return win_rate

def get_mid_price(order_depth: OrderDepth) -> Optional[float]:
    """
    Calculate mid price from order depth.
    
    Args:
        order_depth: OrderDepth object
        
    Returns:
        Mid price or None if not available
    """
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    return None

def calculate_market_impact(order_depth: OrderDepth, quantity: int) -> float:
    """
    Calculate the market impact of executing a trade.
    
    Args:
        order_depth: OrderDepth object
        quantity: Quantity to trade (positive for buy, negative for sell)
        
    Returns:
        Expected market impact in price units
    """
    if quantity == 0:
        return 0.0
    
    # For buys, we need to walk the sell order book
    if quantity > 0:
        if not order_depth.sell_orders:
            return float('inf')  # No liquidity
            
        remaining = quantity
        total_cost = 0
        
        # Sort ask prices in ascending order
        sorted_asks = sorted(order_depth.sell_orders.items())
        
        for price, volume in sorted_asks:
            # Volume is negative in sell orders
            available = abs(volume)
            executed = min(remaining, available)
            total_cost += executed * price
            remaining -= executed
            
            if remaining == 0:
                break
        
        # If we couldn't fill the entire order
        if remaining > 0:
            return float('inf')
            
        # Average execution price
        avg_price = total_cost / quantity
        
        # Market impact = avg_price - best_ask
        best_ask = min(order_depth.sell_orders.keys())
        return avg_price - best_ask
    
    # For sells, we need to walk the buy order book
    else:
        if not order_depth.buy_orders:
            return float('inf')  # No liquidity
            
        remaining = abs(quantity)
        total_revenue = 0
        
        # Sort bid prices in descending order
        sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
        
        for price, volume in sorted_bids:
            executed = min(remaining, volume)
            total_revenue += executed * price
            remaining -= executed
            
            if remaining == 0:
                break
        
        # If we couldn't fill the entire order
        if remaining > 0:
            return float('inf')
            
        # Average execution price
        avg_price = total_revenue / abs(quantity)
        
        # Market impact = best_bid - avg_price
        best_bid = max(order_depth.buy_orders.keys())
        return best_bid - avg_price