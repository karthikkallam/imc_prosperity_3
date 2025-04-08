# datamodel.py
import json
from typing import Dict, List, Any
from json import JSONEncoder
import jsonpickle

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination

    def __repr__(self) -> str:
        return f"(Listing: {self.symbol}, {self.product}, {self.denomination})"


class ConversionObservation:
    # Adjusted based on the Prosperity 3 documentation snippet
    # Note: Actual fields might vary per round. This matches the provided snippet.
    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sunlight: float, humidity: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sunlight = sunlight # Example field from P3 docs
        self.humidity = humidity # Example field from P3 docs

    def __repr__(self) -> str:
        # Provide a more informative representation
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"(ConversionObservation: {attrs})"


class Observation:
    def __init__(self, plainValueObservations: Dict[Product, ObservationValue], conversionObservations: Dict[Product, ConversionObservation]) -> None:
        # Ensure observations are dictionaries even if None is passed
        self.plainValueObservations = plainValueObservations if plainValueObservations is not None else {}
        self.conversionObservations = conversionObservations if conversionObservations is not None else {}

    def __repr__(self) -> str:
        # Use jsonpickle for potentially complex nested structures
        return f"(Observation:\n plainValueObservations: {jsonpickle.encode(self.plainValueObservations)}\n conversionObservations: {jsonpickle.encode(self.conversionObservations)}\n)"

    # Keep __str__ for compatibility if needed, but repr is often more useful for debugging
    def __str__(self) -> str:
         return self.__repr__()


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        # Ensure price is integer
        self.price = int(round(price)) if isinstance(price, float) else int(price)
        self.quantity = int(quantity)

    def __str__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"

    def __repr__(self) -> str:
        return f"Order({self.symbol!r}, {self.price}, {self.quantity})"


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

    def __repr__(self) -> str:
        # Show sorted orders for clarity
        sorted_buys = dict(sorted(self.buy_orders.items(), reverse=True))
        sorted_sells = dict(sorted(self.sell_orders.items()))
        return f"(OrderDepth: Buys: {sorted_buys}, Sells: {sorted_sells})"


class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = "", seller: UserId = "", timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price: int = int(price)
        self.quantity: int = int(quantity)
        # Ensure buyer/seller are always strings, default to empty string if None
        self.buyer = buyer if buyer is not None else ""
        self.seller = seller if seller is not None else ""
        self.timestamp = timestamp

    def __str__(self) -> str:
        # Slightly cleaner string representation
        direction = f"{self.buyer} buys from {self.seller}" if self.buyer else f"{self.seller} sells"
        if self.buyer == "SUBMISSION": direction = "BOT BUYS from US"
        if self.seller == "SUBMISSION": direction = "WE SELL to BOT" # Simplified perspective
        return f"({self.timestamp}@{self.price} {self.quantity}x {self.symbol} [{direction}])"


    def __repr__(self) -> str:
        # More explicit representation
        return f"Trade(t={self.timestamp}, sym={self.symbol!r}, p={self.price}, q={self.quantity}, B={self.buyer!r}, S={self.seller!r})"


class TradingState(object):
    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings if listings is not None else {}
        self.order_depths = order_depths if order_depths is not None else {}
        self.own_trades = own_trades if own_trades is not None else {}
        self.market_trades = market_trades if market_trades is not None else {}
        # Ensure position is always a dictionary
        self.position = position if position is not None else {}
        # Ensure observations is always an Observation object
        self.observations = observations if observations is not None else Observation({}, {})

    def toJSON(self):
        # Using ProsperityEncoder helps if we add custom objects later
        # Default lambda works fine for standard types and __dict__-able objects
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, cls=ProsperityEncoder)

    def __repr__(self) -> str:
        # Provide a summarized representation for easier debugging
        return (f"TradingState(t={self.timestamp}, Pos={self.position}, "
                f"Orders={len(self.order_depths)}, OwnTrades={sum(len(v) for v in self.own_trades.values())}, "
                f"MktTrades={sum(len(v) for v in self.market_trades.values())}, "
                f"Obs={self.observations != Observation({}, {})}, TraderDataLen={len(self.traderData)})")


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, (OrderDepth, Trade, Order, Listing, Observation, ConversionObservation)):
             # Use __dict__ for our custom classes if they don't have specific encoding needs yet
             return o.__dict__
        # Add handling for other complex types (like numpy arrays if used) here if necessary
        # For numpy types:
        # if isinstance(o, np.integer): return int(o)
        # if isinstance(o, np.floating): return float(o)
        # if isinstance(o, np.ndarray): return o.tolist()
        try:
            # Fallback to default JSON encoding for basic types
            return JSONEncoder.default(self, o)
        except TypeError:
            # If still failing, return string representation as a last resort
             return str(o)