# trader.py
import json
import jsonpickle
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Import necessary classes from datamodel.py
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# --- Logger Class (using provided one, assuming it's robust) ---
# (Keep your Logger class exactly as provided in the prompt)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3950 # Adjusted based on observations

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        log_message = sep.join(map(str, objects)) + end
        if len(self.logs) + len(log_message) <= self.max_log_length:
            self.logs += log_message
        # else: # Optional: Log truncation silently
        #     if not self.logs.endswith("...\nLOG TRUNCATED\n"):
        #          self.logs += "...\nLOG TRUNCATED\n"

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Simple flush: print directly. Ensure logger output doesn't exceed overall limits.
        # The platform likely combines stdout from print() with the return tuple.
        # Keep logs concise.
        print(self.logs)
        self.logs = "" # Clear logs for the next iteration

    # --- Minimalist Compression/Truncation for TraderData/Logs if needed ---
    # These are simple placeholders if the logger needs explicit limits before flushing
    def to_json(self, value: Any) -> str:
        return jsonpickle.encode(value, unpicklable=False)

    def truncate(self, value: str | None, max_length: int) -> str:
        s_value = str(value) if value is not None else ""
        if len(s_value) <= max_length:
            return s_value
        return s_value[:max_length - 3] + "..."

# Global logger instance
logger = Logger()


# --- Product Constants ---
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"


# --- Default Parameters ---
# These will be overridden by Optuna during optimization
# Added EMA windows and skew factors
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "anchor_value": 10000, # Strong prior fair value
        "use_ema": True,        # Whether to use EMA adjustment
        "ema_alpha": 0.15,      # Smoothing factor for EMA (adjusts sensitivity to recent prices)
        "take_width": 1,        # Take aggressively if price deviates by this much from fair
        "clear_width": 0,       # How far beyond fair value to trade when clearing inventory
        # Making parameters
        "disregard_edge": 1,    # Ignore book levels within fair +/- this when pennying/joining
        "join_edge": 2,         # Max distance from fair to join existing level
        "default_edge": 3,      # Default spread width if not joining/penying
        "skew_factor": 0.03,    # How much to skew quotes per unit of position ratio (0=off)
        "soft_limit_ratio": 0.7 # Position ratio (pos/limit) above which skew starts aggressively
    },
    Product.KELP: {
        "use_ema": True,
        "ema_alpha": 0.3,       # KELP might need faster EMA
        "reversion_beta": -0.2, # Mean reversion strength (NEEDS TUNING)
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,# Avoid taking large passive orders
        "adverse_volume": 15,   # Volume threshold for adverse selection filter
        # Making parameters
        "disregard_edge": 1,
        "join_edge": 0,         # Never join for KELP (strategy choice)
        "default_edge": 1,      # Tight default spread for KELP
        "skew_factor": 0.05,    # KELP might need stronger skew
        "soft_limit_ratio": 0.6
    },
}

# Type alias for Trader Data state
TraderStateData = Dict[str, Any]

class Trader:
    def __init__(self, params: Optional[Dict] = None) -> None:
        """
        Initializes the Trader.
        Args:
            params (Dict | None, optional): Override parameters. Defaults to PARAMS.
        """
        # Use provided params or default PARAMS
        self.params = params if params is not None else PARAMS

        # --- !! POSITION LIMITS !! ---
        # These limits strongly influence the maximum achievable PnL.
        # If Optuna gets stuck, it might be because the strategy consistently
        # hits these limits for the tested round (e.g., Round 0).
        # Consider if these limits can be changed based on competition rules,
        # or if the strategy needs fundamental changes to work better within them.
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50}
        # --- !! POSITION LIMITS !! ---


        # Validate PARAMS structure
        for prod, p_dict in self.params.items():
            if prod not in self.LIMIT:
                logger.print(f"WARN: Product '{prod}' in PARAMS but not in LIMITS {self.LIMIT.keys()}.")
            # Add more checks if needed (e.g., required keys)

        logger.print("Trader Initialized with PARAMS:", jsonpickle.encode(self.params))
        logger.print("Trader Initialized with LIMITS:", self.LIMIT)

    # --- Helper Functions ---
    def _get_position(self, product: str, state: TradingState) -> int:
        return state.position.get(product, 0)

    def _get_order_book(self, product: str, state: TradingState) -> OrderDepth:
        return state.order_depths.get(product, OrderDepth()) # Return empty book if not present

    def _get_best_bid(self, order_depth: OrderDepth) -> Optional[int]:
        return max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    def _get_best_ask(self, order_depth: OrderDepth) -> Optional[int]:
        return min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    def _get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = self._get_best_bid(order_depth)
        best_ask = self._get_best_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None # Not calculable

    def _update_ema(self, current_value: Optional[float], prev_ema: Optional[float], alpha: float) -> Optional[float]:
        """Updates the Exponential Moving Average."""
        if current_value is None:
            return prev_ema # No new data, keep old EMA
        if prev_ema is None:
            return current_value # Initialize EMA with first value
        return alpha * current_value + (1 - alpha) * prev_ema

    def _load_trader_data(self, state: TradingState) -> TraderStateData:
        """Safely loads trader data from the state."""
        if state.traderData is not None and state.traderData != "":
            try:
                decoded_data = jsonpickle.decode(str(state.traderData))
                if isinstance(decoded_data, dict):
                    return decoded_data
                else:
                    logger.print(f"WARN: Decoded traderData type {type(decoded_data)}, expected dict. Initializing.")
                    return {}
            except Exception as e:
                logger.print(f"ERROR decoding traderData ('{state.traderData}'). Error: {e}. Initializing.")
                return {}
        return {} # Initialize fresh if empty

    def _save_trader_data(self, trader_data: TraderStateData) -> str:
        """Safely saves trader data."""
        try:
            # Ensure only JSON-serializable types are included for safety
            serializable_data = {k: v for k, v in trader_data.items() if isinstance(v, (str, int, float, list, dict, bool, type(None)))}
            # Limit the length of the saved data string if necessary
            encoded = jsonpickle.encode(serializable_data, unpicklable=False)
            # MAX_TRADER_DATA_LEN = 2000 # Example limit if needed
            # if len(encoded) > MAX_TRADER_DATA_LEN:
            #     logger.print(f"WARN: Trader data exceeds limit ({len(encoded)} > {MAX_TRADER_DATA_LEN}), potential truncation needed.")
            #     # Implement truncation logic here if required
            return encoded
        except Exception as e:
             logger.print(f"ERROR encoding final traderData: {trader_data}. Error: {e}")
             return "{}" # Use empty JSON object as fallback

    # --- Fair Value Calculation ---
    def calculate_fair_value(self, product: str, order_depth: OrderDepth, trader_data: TraderStateData) -> Optional[float]:
        """Calculates the fair value for a given product."""
        params = self.params.get(product)
        if not params:
            logger.print(f"ERROR: No parameters found for {product}")
            return None

        mid_price = self._get_mid_price(order_depth)
        fair_value = mid_price # Default to mid if no other logic applies or fails

        # --- RAINFOREST RESIN ---
        if product == Product.RAINFOREST_RESIN:
            anchor = params.get('anchor_value', 10000)
            fair_value = float(anchor) # Start with the anchor
            if params.get('use_ema', False):
                ema_key = f"{product}_ema"
                prev_ema = trader_data.get(ema_key)
                alpha = params.get('ema_alpha', 0.1)
                # Update EMA using mid-price, but perhaps cap deviation from anchor
                current_val_for_ema = mid_price
                # Optional: Limit how far the mid_price can pull the EMA from the anchor
                # if mid_price is not None:
                #     max_dev = 5 # Example: Allow EMA to drift +/- 5 from anchor
                #     current_val_for_ema = max(anchor - max_dev, min(anchor + max_dev, mid_price))

                updated_ema = self._update_ema(current_val_for_ema, prev_ema, alpha)
                trader_data[ema_key] = updated_ema
                if updated_ema is not None:
                    fair_value = updated_ema # Use EMA if available
                else:
                    fair_value = float(anchor) # Fallback to anchor if EMA fails
            # logger.print(f"{product} Fair Value (Anchor: {anchor}, EMA Used: {params.get('use_ema', False)}): {fair_value:.2f}")


        # --- KELP ---
        elif product == Product.KELP:
            prev_ema = trader_data.get(f"{product}_ema")
            prev_fair_value = trader_data.get(f"{product}_last_fair", mid_price) # Use last known fair value as fallback

            if params.get('use_ema', False):
                alpha = params.get('ema_alpha', 0.2)
                # Update EMA with the current mid-price
                current_ema = self._update_ema(mid_price, prev_ema, alpha)
                trader_data[f"{product}_ema"] = current_ema

                if current_ema is not None:
                    # Apply Mean Reversion based on EMA changes
                    beta = params.get('reversion_beta', 0.0)
                    if prev_ema is not None and prev_ema != 0 and beta != 0:
                        try:
                             ema_return = (current_ema - prev_ema) / prev_ema
                             # Optional: Bound the return to prevent extreme predictions
                             max_abs_return = 0.05 # Limit to +/- 5% change prediction
                             bounded_return = max(min(ema_return, max_abs_return), -max_abs_return)

                             pred_return = bounded_return * beta
                             fair_value = current_ema * (1 + pred_return)
                             # logger.print(f"KELP Reversion: EMA={current_ema:.2f}, PrevEMA={prev_ema:.2f}, Ret={bounded_return:.4f}, PredRet={pred_return:.4f}, Fair={fair_value:.2f}")
                        except Exception as e:
                             logger.print(f"WARN: Error during KELP mean reversion: {e}. Using EMA {current_ema:.2f}.")
                             fair_value = current_ema
                    else:
                        # Not enough history or beta=0, use current EMA as fair value
                        fair_value = current_ema
                        # logger.print(f"INFO: KELP using EMA {fair_value:.2f} as fair value (No reversion: PrevEMA={prev_ema}, Beta={beta}).")
                else:
                    # EMA calculation failed (e.g., no mid_price), use last known fair value
                    fair_value = prev_fair_value
                    # logger.print(f"WARN: KELP using previous fair value {fair_value:.2f} due to EMA failure.")
            else:
                # Not using EMA, revert to simple mid-price or last known value
                fair_value = mid_price if mid_price is not None else prev_fair_value
                # logger.print(f"INFO: KELP using {'mid-price' if mid_price is not None else 'previous fair value'} {fair_value:.2f} (EMA disabled).")

            # Store the calculated fair value for the next iteration's fallback
            trader_data[f"{product}_last_fair"] = fair_value

        else:
            # Fallback for unknown products (use mid-price)
            fair_value = mid_price
            logger.print(f"WARN: Using simple mid-price {fair_value} for unknown product {product}")

        # Ensure fair_value is float or None
        return float(fair_value) if fair_value is not None else None


    # --- Order Placement Logic ---

    def take_best_orders(
        self, product: str, fair_value: float, orders: List[Order],
        order_depth: OrderDepth, position: int, buy_vol_committed: int, sell_vol_committed: int
    ) -> Tuple[int, int]:
        """Places orders to take favorable prices."""
        params = self.params[product]
        take_width = params['take_width']
        position_limit = self.LIMIT[product]
        prevent_adverse = params.get("prevent_adverse", False)
        adverse_volume = params.get("adverse_volume", 0)

        # Available capacity NOW, considering current position and orders *already committed* this step
        current_buy_capacity = position_limit - (position + buy_vol_committed)
        current_sell_capacity = position_limit + (position - sell_vol_committed) # Positive value

        # --- Take Asks (Execute Buy Orders) ---
        best_ask = self._get_best_ask(order_depth)
        if best_ask is not None and best_ask <= fair_value - take_width:
            # Check if we have capacity to buy
            if current_buy_capacity > 0:
                best_ask_amount = abs(order_depth.sell_orders[best_ask])
                vol_to_take = min(best_ask_amount, current_buy_capacity)

                execute_trade = True
                # Check for adverse selection (large volume potentially moving the market)
                if prevent_adverse and best_ask_amount > adverse_volume:
                     execute_trade = False
                     # logger.print(f"TAKE AVOID (BUY {product}): Vol {best_ask_amount}@{best_ask} > adverse {adverse_volume}")

                if execute_trade and vol_to_take > 0:
                    logger.print(f"EXECUTE TAKE BUY {product}: {vol_to_take}@{best_ask} (Fair: {fair_value:.2f}, Cap: {current_buy_capacity})")
                    orders.append(Order(product, best_ask, vol_to_take))
                    buy_vol_committed += vol_to_take # IMPORTANT: Update committed volume

        # --- Take Bids (Execute Sell Orders) ---
        # Recalculate sell capacity as buy orders might have affected position implicitly
        current_sell_capacity = position_limit + (position - sell_vol_committed + buy_vol_committed) # Adjust for buys just added

        best_bid = self._get_best_bid(order_depth)
        if best_bid is not None and best_bid >= fair_value + take_width:
             # Check if we have capacity to sell
             if current_sell_capacity > 0:
                best_bid_amount = abs(order_depth.buy_orders[best_bid])
                vol_to_take = min(best_bid_amount, current_sell_capacity)

                execute_trade = True
                # Check for adverse selection
                if prevent_adverse and best_bid_amount > adverse_volume:
                     execute_trade = False
                     # logger.print(f"TAKE AVOID (SELL {product}): Vol {best_bid_amount}@{best_bid} > adverse {adverse_volume}")

                if execute_trade and vol_to_take > 0:
                    logger.print(f"EXECUTE TAKE SELL {product}: {vol_to_take}@{best_bid} (Fair: {fair_value:.2f}, Cap: {current_sell_capacity})")
                    orders.append(Order(product, best_bid, -vol_to_take))
                    sell_vol_committed += vol_to_take # IMPORTANT: Update committed volume

        return buy_vol_committed, sell_vol_committed


    def clear_position_order(
        self, product: str, fair_value: float, orders: List[Order],
        order_depth: OrderDepth, position: int, buy_vol_committed: int, sell_vol_committed: int
    ) -> Tuple[int, int]:
        """Places orders to aggressively clear inventory towards fair value."""
        params = self.params[product]
        clear_width = params['clear_width']
        position_limit = self.LIMIT[product]

        # Position *after* any take orders decided in this step
        position_after_take = position + buy_vol_committed - sell_vol_committed

        if position_after_take == 0:
            return buy_vol_committed, sell_vol_committed # Nothing to clear

        # How much capacity remains *after* take orders for further trades this step
        # Note: Capacity calculation here needs careful review - does it overlap with market_make?
        # Assuming these clear orders consume capacity needed later for making.
        remaining_buy_capacity = position_limit - (position + buy_vol_committed)
        remaining_sell_capacity = position_limit + (position - sell_vol_committed) # Recalc needed? Let's assume simple for now

        # --- Clear LONG position (place SELL orders) ---
        if position_after_take > 0:
            qty_to_clear = position_after_take
            qty_cleared = 0
            # Sell at bids AT or ABOVE fair_value + clear_width
            clear_sell_price_threshold = math.ceil(fair_value + clear_width)

            # Iterate through available bids from best to worst
            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
            for price, available_vol in sorted_bids:
                # Stop if cleared enough or no more sell capacity (respecting limit)
                current_sell_cap_check = position_limit + (position + buy_vol_committed - sell_vol_committed)
                if qty_cleared >= qty_to_clear or current_sell_cap_check <= 0: break

                if price >= clear_sell_price_threshold:
                    vol_needed = qty_to_clear - qty_cleared
                    # Trade volume is limited by need, available market vol, and remaining capacity to sell
                    trade_vol = min(vol_needed, available_vol, current_sell_cap_check)
                    if trade_vol > 0:
                        logger.print(f"EXECUTE CLEAR LONG {product}: Selling {trade_vol}@{price} (Fair: {fair_value:.2f}, Threshold >= {clear_sell_price_threshold}, Cap: {current_sell_cap_check})")
                        orders.append(Order(product, price, -trade_vol))
                        sell_vol_committed += trade_vol
                        qty_cleared += trade_vol
                        # Update remaining capacity implicitly tracked by sell_vol_committed
                else: break # Prices sorted, won't find better bids

        # --- Clear SHORT position (place BUY orders) ---
        elif position_after_take < 0:
            qty_to_clear = abs(position_after_take)
            qty_cleared = 0
            # Buy at asks AT or BELOW fair_value - clear_width
            clear_buy_price_threshold = math.floor(fair_value - clear_width)

            # Iterate through available asks from best to worst
            sorted_asks = sorted(order_depth.sell_orders.items())
            for price, available_vol_neg in sorted_asks:
                # Stop if cleared enough or no more buy capacity (respecting limit)
                current_buy_cap_check = position_limit - (position + buy_vol_committed - sell_vol_committed)
                if qty_cleared >= qty_to_clear or current_buy_cap_check <= 0: break

                if price <= clear_buy_price_threshold:
                    available_vol = abs(available_vol_neg)
                    vol_needed = qty_to_clear - qty_cleared
                    # Trade volume is limited by need, available market vol, and remaining capacity to buy
                    trade_vol = min(vol_needed, available_vol, current_buy_cap_check)
                    if trade_vol > 0:
                        logger.print(f"EXECUTE CLEAR SHORT {product}: Buying {trade_vol}@{price} (Fair: {fair_value:.2f}, Threshold <= {clear_buy_price_threshold}, Cap: {current_buy_cap_check})")
                        orders.append(Order(product, price, trade_vol))
                        buy_vol_committed += trade_vol
                        qty_cleared += trade_vol
                         # Update remaining capacity implicitly tracked by buy_vol_committed
                else: break # Prices sorted, won't find better asks

        return buy_vol_committed, sell_vol_committed


    def calculate_making_quotes(
        self, product: str, fair_value: float, order_depth: OrderDepth,
        position: int, buy_vol_committed: int, sell_vol_committed: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """ Calculates the bid and ask prices for market making orders, including skew."""
        params = self.params[product]
        position_limit = self.LIMIT[product]
        disregard_edge = params["disregard_edge"]
        join_edge = params["join_edge"]
        default_edge = params["default_edge"]
        skew_factor = params.get("skew_factor", 0.0)
        soft_limit_ratio = params.get("soft_limit_ratio", 1.0) # Ratio at which skew becomes more aggressive

        # Effective position *after* take/clear orders decided this step
        effective_position = position + buy_vol_committed - sell_vol_committed
        position_ratio = effective_position / position_limit if position_limit != 0 else 0

        # --- Calculate Base Bid/Ask (Pennying/Joining Logic) ---
        asks_outside_disregard = {p: v for p, v in order_depth.sell_orders.items() if p > fair_value + disregard_edge}
        bids_outside_disregard = {p: v for p, v in order_depth.buy_orders.items() if p < fair_value - disregard_edge}
        best_ask_outside = min(asks_outside_disregard.keys()) if asks_outside_disregard else None
        best_bid_outside = max(bids_outside_disregard.keys()) if bids_outside_disregard else None

        base_ask = fair_value + default_edge
        if best_ask_outside is not None:
            # Join if the best outside ask is close enough
            if join_edge > 0 and (best_ask_outside - fair_value) <= join_edge:
                base_ask = best_ask_outside  # Join
            else: # Otherwise, try to penny inside it
                base_ask = best_ask_outside - 1 # Penny
                # Ensure pennying doesn't cross fair value
                if base_ask <= fair_value: base_ask = fair_value + 1

        base_bid = fair_value - default_edge
        if best_bid_outside is not None:
            # Join if the best outside bid is close enough
            if join_edge > 0 and (fair_value - best_bid_outside) <= join_edge:
                base_bid = best_bid_outside # Join
            else: # Otherwise, try to penny inside it
                base_bid = best_bid_outside + 1 # Penny
                 # Ensure pennying doesn't cross fair value
                if base_bid >= fair_value: base_bid = fair_value - 1

        # Ensure base bid < base ask (minimum spread of 2 if they meet/cross)
        if base_bid >= base_ask:
            base_bid = fair_value - 1 # Fallback spread
            base_ask = fair_value + 1

        # --- Apply Position Skew ---
        skew_amount = 0
        if skew_factor > 0 and position_limit > 0:
            # Skew more aggressively when closer to limits (e.g., beyond soft_limit_ratio)
            abs_position_ratio = abs(position_ratio)
            skew_magnitude = abs_position_ratio
            if abs_position_ratio > soft_limit_ratio:
                 # Example: Linear increase in skew effect past soft limit
                 overshoot = abs_position_ratio - soft_limit_ratio
                 # Scale factor > 1 determines how much more aggressive skew becomes
                 skew_aggressiveness_factor = 2.0
                 skew_magnitude = soft_limit_ratio + overshoot * skew_aggressiveness_factor

            # Calculate skew amount (e.g., proportional to default edge)
            # Adjust scale factor carefully based on tuning/observation
            skew_scale_factor = default_edge # Or maybe a fixed value like 1 or 2?
            skew_amount = skew_scale_factor * skew_factor * skew_magnitude

            # Apply skew: If long (pos_ratio > 0), lower both bid and ask to encourage selling.
            # If short (pos_ratio < 0), raise both bid and ask to encourage buying.
            if position_ratio > 0: # Long
                skew_amount = -abs(skew_amount)
                # logger.print(f"SKEW LONG {product}: PosRatio={position_ratio:.2f}, SkewAdj={skew_amount:.2f}")
            elif position_ratio < 0: # Short
                skew_amount = abs(skew_amount)
                # logger.print(f"SKEW SHORT {product}: PosRatio={position_ratio:.2f}, SkewAdj={skew_amount:.2f}")

        final_bid = base_bid + skew_amount
        final_ask = base_ask + skew_amount

        # --- Final Rounding and Sanity Checks ---
        final_bid_int = math.floor(final_bid)
        final_ask_int = math.ceil(final_ask)

        # Ensure bid < ask by at least 1 tick AFTER skew and rounding
        if final_bid_int >= final_ask_int:
             # If skew caused overlap, adjust minimally around the *skewed* fair value
             skewed_fair_value = fair_value + skew_amount
             final_bid_int = math.floor(skewed_fair_value - 0.5) # Bid slightly below skewed mid
             final_ask_int = math.ceil(skewed_fair_value + 0.5)  # Ask slightly above skewed mid
             # Ensure they are still different after rounding
             if final_bid_int >= final_ask_int:
                  final_ask_int = final_bid_int + 1 # Force 1 tick spread
             # logger.print(f"WARN {product}: Adjusted bid/ask clash after skew. New Bid: {final_bid_int}, Ask: {final_ask_int}")

        # Optional: Prevent quotes too far from fair value
        # max_quote_dev = default_edge * 3
        # if abs(final_bid_int - fair_value) > max_quote_dev or abs(final_ask_int - fair_value) > max_quote_dev:
        #     logger.print(f"WARN {product}: Quotes ({final_bid_int}, {final_ask_int}) too far from fair ({fair_value:.2f}). Skipping make.")
        #     return None, None

        return final_bid_int, final_ask_int


    def market_make(
        self, product: str, orders: List[Order], bid_price: Optional[int], ask_price: Optional[int],
        position: int, buy_vol_committed: int, sell_vol_committed: int
    ) -> None:
        """Places making bid and ask orders based on remaining capacity."""
        if bid_price is None or ask_price is None:
             # logger.print(f"INFO {product}: Skipping make order placement (invalid prices).")
             return # Cannot make if prices aren't calculable

        position_limit = self.LIMIT[product]

        # Calculate final capacity *after* considering trades from take/clear steps
        effective_position = position + buy_vol_committed - sell_vol_committed
        final_buy_capacity = position_limit - effective_position
        final_sell_capacity = position_limit + effective_position # Capacity to sell is limit + current pos

        # Place buy quote if there's capacity
        if final_buy_capacity > 0:
            # logger.print(f"MAKE BUY {product}: {final_buy_capacity}@{bid_price}")
            orders.append(Order(product, bid_price, final_buy_capacity))
        # else: logger.print(f"INFO {product}: No capacity to place MAKE BUY (Limit: {position_limit}, EffPos: {effective_position})")


        # Place sell quote if there's capacity
        if final_sell_capacity > 0:
            # logger.print(f"MAKE SELL {product}: {final_sell_capacity}@{ask_price}")
            orders.append(Order(product, ask_price, -final_sell_capacity))
        # else: logger.print(f"INFO {product}: No capacity to place MAKE SELL (Limit: {position_limit}, EffPos: {effective_position})")



    # --- Main Execution Method ---

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic method called by the simulation environment.
        """
        logger.print(f"\n========== Trader Run: Timestamp {state.timestamp} ==========")

        # Load previous state
        trader_data = self._load_trader_data(state)

        # Initialize results dictionary
        result: Dict[str, List[Order]] = {}

        # --- Process each product ---
        products_to_process = list(self.params.keys()) # Get products from params
        if not products_to_process: # Check if params dict might be empty
             products_to_process = list(state.order_depths.keys()) # Fallback to available market data

        for product in products_to_process:
            # Ensure product has parameters AND is in the current market state
            if product not in self.params or product not in state.order_depths or product not in self.LIMIT:
                # logger.print(f"INFO: Skipping product '{product}' - missing params, market data, or LIMITS.")
                continue

            logger.print(f"--- Processing {product} (Pos: {self._get_position(product, state)}) ---")
            order_depth = self._get_order_book(product, state)
            position = self._get_position(product, state)
            orders_for_product: List[Order] = []

            # Track cumulative volume committed *within this timestamp* for limit checks
            # Reset for each product
            buy_volume_committed_this_step = 0
            sell_volume_committed_this_step = 0

            # --- Calculate Fair Value ---
            fair_value = self.calculate_fair_value(product, order_depth, trader_data)

            if fair_value is None:
                logger.print(f"ERROR: Fair value calculation failed for {product}. Skipping orders.")
                result[product] = []
                continue # Skip to next product

            logger.print(f"{product} Fair Value: {fair_value:.2f}")

            # --- Order Generation Sequence ---
            # Order matters: Take opportunistic trades first, then clear unwanted inventory, then place passive quotes.

            # 1. Take Orders (Hit Bids/Asks beyond take_width)
            buy_volume_committed_this_step, sell_volume_committed_this_step = self.take_best_orders(
                product, fair_value, orders_for_product, order_depth, position,
                buy_volume_committed_this_step, sell_volume_committed_this_step
            )
            # logger.print(f"{product} After Take: BuyVol={buy_volume_committed_this_step}, SellVol={sell_volume_committed_this_step}")

            # 2. Clear Position Orders (Aggressively trade towards zero if needed)
            buy_volume_committed_this_step, sell_volume_committed_this_step = self.clear_position_order(
                product, fair_value, orders_for_product, order_depth, position,
                buy_volume_committed_this_step, sell_volume_committed_this_step
            )
            # logger.print(f"{product} After Clear: BuyVol={buy_volume_committed_this_step}, SellVol={sell_volume_committed_this_step}")


            # 3. Calculate Making Quotes (Penny/Join logic + Skew based on effective position)
            make_bid, make_ask = self.calculate_making_quotes(
                 product, fair_value, order_depth, position,
                 buy_volume_committed_this_step, sell_volume_committed_this_step # Pass committed vols
            )
            # logger.print(f"{product} Calculated Make Quotes: Bid={make_bid}, Ask={make_ask}")


            # 4. Place Making Orders (using remaining capacity AFTER take/clear)
            self.market_make(
                product, orders_for_product, make_bid, make_ask, position,
                buy_volume_committed_this_step, sell_volume_committed_this_step # Pass committed vols
            )

            # Assign the collected orders to the result dictionary
            result[product] = orders_for_product
            # if orders_for_product: # Log only if orders were generated
            #      logger.print(f"-> {product} Final Orders ({len(result[product])}): {orders_for_product}")


        # --- State Persistence & Return ---
        final_trader_data_str = self._save_trader_data(trader_data)

        # Conversions: Not applicable in tutorial
        conversions = 0

        # Flush logs (prints the accumulated log messages)
        # The platform combines stdout (from logger.flush) and the return tuple.
        logger.flush(state, result, conversions, final_trader_data_str)

        # Return orders, conversions, and trader state string
        return result, conversions, final_trader_data_str