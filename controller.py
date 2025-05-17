# controller.py
import os
import json
from datetime import datetime

from modules.data_collection import DataCollector
from modules.ai_prediction_engine import PredictionEngine
from modules.trading_strategy_module import TrendFollowingStrategy
from modules.risk_management_module import RiskManager
from modules.order_management_system import OrderSide
from modules.angel_one_integration import AngelOneAPI, OrderParams

# Configurations
CONFIG_PATH = "config/risk_config.json"
SYMBOLS = ["RELIANCE-EQ", "INFY-EQ", "TCS-EQ"]
DEFAULT_ACCOUNT_BALANCE = 1000000

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_full_pipeline(symbols=SYMBOLS, live=False, risk=1.0):
    results = []

    # Modules
    collector = DataCollector()
    predictor = PredictionEngine()
    strategy = TrendFollowingStrategy(symbols)
    risk_manager = RiskManager(CONFIG_PATH)
    broker = AngelOneAPI() if live else None

    account_info = {"balance": DEFAULT_ACCOUNT_BALANCE}

    for symbol in symbols:
        df = collector.fetch_stock_data(symbol, period="1mo", interval="1d")
        if df is None or df.empty:
            continue
        df = collector.add_technical_indicators(df)

        # Dummy prediction (replace with model prediction)
        prediction = {
            symbol: {
                "direction": 1,
                "confidence": 0.8,
                "model_name": "mock_model"
            }
        }
        market_data = {
            symbol: {
                "close": df["Close"].tolist()
            }
        }

        signals = strategy.generate_signals(prediction, market_data)
        trades = strategy.convert_to_trades(signals, account_info)

        for trade in trades:
            trade_dict = trade.to_order_dict()

            approved, reason = risk_manager.check_trade({
                "symbol": trade.symbol,
                "direction": trade.side.value,
                "quantity": trade.quantity,
                "price": trade_dict.get("price", 100),
                "account_value": account_info["balance"]
            })

            if not approved:
                results.append({"symbol": trade.symbol, "status": "REJECTED", "reason": reason})
                continue

            if live:
                order = OrderParams(
                    symbol=trade.symbol,
                    quantity=int(trade.quantity),
                    side=trade.side.value,
                    order_type=trade.order_type.value,
                    product_type="INTRADAY",
                    price=trade.price or 0,
                    stoploss=trade.stop_loss or 0,
                    target=trade.take_profit or 0
                )
                response = broker.place_order(order)
                results.append({"symbol": trade.symbol, "status": "LIVE_ORDER", "details": response})
            else:
                results.append({"symbol": trade.symbol, "status": "PAPER_ORDER", "details": trade_dict})

    return results

if __name__ == "__main__":
    import pprint
    pprint.pprint(run_full_pipeline(live=False))
