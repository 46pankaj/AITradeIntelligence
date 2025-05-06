import logging
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingJournal:
    """
    Class for managing a trading journal and generating trading statistics
    """
    def __init__(self):
        # Journal entries
        self.trade_entries = []
        self.strategy_entries = []
        self.market_entries = []
        self.notes = []
        
        # Settings
        self.max_entries = 1000  # Maximum number of entries to keep
        
        # Load saved data
        self._load_data()
        
    def add_trade(self, trade_data):
        """
        Add a trade to the journal
        
        Args:
            trade_data (dict): Trade data
            
        Returns:
            str: Entry ID
        """
        # Generate ID
        entry_id = f"TRADE-{len(self.trade_entries)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add timestamp if not present
        if "timestamp" not in trade_data:
            trade_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Add ID
        trade_data["id"] = entry_id
        
        # Add to journal
        self.trade_entries.append(trade_data)
        
        # Trim if needed
        if len(self.trade_entries) > self.max_entries:
            self.trade_entries = self.trade_entries[-self.max_entries:]
            
        logger.info(f"Added trade to journal: {trade_data.get('symbol')} {trade_data.get('type')} (ID: {entry_id})")
        self._save_data()
        
        return entry_id
        
    def add_strategy(self, strategy_data):
        """
        Add a strategy event to the journal
        
        Args:
            strategy_data (dict): Strategy data
            
        Returns:
            str: Entry ID
        """
        # Generate ID
        entry_id = f"STRAT-{len(self.strategy_entries)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add timestamp if not present
        if "timestamp" not in strategy_data:
            strategy_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Add ID
        strategy_data["id"] = entry_id
        
        # Add to journal
        self.strategy_entries.append(strategy_data)
        
        # Trim if needed
        if len(self.strategy_entries) > self.max_entries:
            self.strategy_entries = self.strategy_entries[-self.max_entries:]
            
        logger.info(f"Added strategy to journal: {strategy_data.get('name')} for {strategy_data.get('symbol')} (ID: {entry_id})")
        self._save_data()
        
        return entry_id
        
    def add_market_analysis(self, market_data):
        """
        Add a market analysis event to the journal
        
        Args:
            market_data (dict): Market analysis data
            
        Returns:
            str: Entry ID
        """
        # Generate ID
        entry_id = f"MARKET-{len(self.market_entries)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add timestamp if not present
        if "timestamp" not in market_data:
            market_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Add ID
        market_data["id"] = entry_id
        
        # Add to journal
        self.market_entries.append(market_data)
        
        # Trim if needed
        if len(self.market_entries) > self.max_entries:
            self.market_entries = self.market_entries[-self.max_entries:]
            
        logger.info(f"Added market analysis to journal: {market_data.get('type')} (ID: {entry_id})")
        self._save_data()
        
        return entry_id
        
    def add_note(self, note_text, tags=None, related_ids=None):
        """
        Add a note to the journal
        
        Args:
            note_text (str): Note text
            tags (list, optional): List of tags
            related_ids (list, optional): List of related entry IDs
            
        Returns:
            str: Entry ID
        """
        # Generate ID
        entry_id = f"NOTE-{len(self.notes)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create note entry
        note = {
            "id": entry_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": note_text,
            "tags": tags or [],
            "related_ids": related_ids or []
        }
        
        # Add to journal
        self.notes.append(note)
        
        # Trim if needed
        if len(self.notes) > self.max_entries:
            self.notes = self.notes[-self.max_entries:]
            
        logger.info(f"Added note to journal (ID: {entry_id})")
        self._save_data()
        
        return entry_id
        
    def get_trades(self, limit=None, symbol=None, trade_type=None, date_from=None, date_to=None):
        """
        Get trades from the journal
        
        Args:
            limit (int, optional): Maximum number of trades to return
            symbol (str, optional): Filter by symbol
            trade_type (str, optional): Filter by trade type (BUY, SELL)
            date_from (str, optional): Filter by date (format: YYYY-MM-DD)
            date_to (str, optional): Filter by date (format: YYYY-MM-DD)
            
        Returns:
            list: List of trades
        """
        # Make a copy of trades
        trades = self.trade_entries.copy()
        
        # Filter by symbol
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]
            
        # Filter by trade type
        if trade_type:
            trades = [t for t in trades if t.get("type") == trade_type]
            
        # Filter by date
        if date_from:
            try:
                date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                trades = [t for t in trades if datetime.strptime(t.get("timestamp").split()[0], "%Y-%m-%d") >= date_from_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_from: {date_from}")
                
        if date_to:
            try:
                date_to_dt = datetime.strptime(date_to, "%Y-%m-%d")
                trades = [t for t in trades if datetime.strptime(t.get("timestamp").split()[0], "%Y-%m-%d") <= date_to_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_to: {date_to}")
                
        # Sort by timestamp (newest first)
        trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit:
            trades = trades[:limit]
            
        return trades
        
    def get_strategies(self, limit=None, symbol=None, date_from=None, date_to=None):
        """
        Get strategies from the journal
        
        Args:
            limit (int, optional): Maximum number of strategies to return
            symbol (str, optional): Filter by symbol
            date_from (str, optional): Filter by date (format: YYYY-MM-DD)
            date_to (str, optional): Filter by date (format: YYYY-MM-DD)
            
        Returns:
            list: List of strategies
        """
        # Make a copy of strategies
        strategies = self.strategy_entries.copy()
        
        # Filter by symbol
        if symbol:
            strategies = [s for s in strategies if s.get("symbol") == symbol]
            
        # Filter by date
        if date_from:
            try:
                date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                strategies = [s for s in strategies if datetime.strptime(s.get("timestamp").split()[0], "%Y-%m-%d") >= date_from_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_from: {date_from}")
                
        if date_to:
            try:
                date_to_dt = datetime.strptime(date_to, "%Y-%m-%d")
                strategies = [s for s in strategies if datetime.strptime(s.get("timestamp").split()[0], "%Y-%m-%d") <= date_to_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_to: {date_to}")
                
        # Sort by timestamp (newest first)
        strategies.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit:
            strategies = strategies[:limit]
            
        return strategies
        
    def get_market_analyses(self, limit=None, analysis_type=None, date_from=None, date_to=None):
        """
        Get market analyses from the journal
        
        Args:
            limit (int, optional): Maximum number of analyses to return
            analysis_type (str, optional): Filter by analysis type
            date_from (str, optional): Filter by date (format: YYYY-MM-DD)
            date_to (str, optional): Filter by date (format: YYYY-MM-DD)
            
        Returns:
            list: List of market analyses
        """
        # Make a copy of market analyses
        analyses = self.market_entries.copy()
        
        # Filter by analysis type
        if analysis_type:
            analyses = [a for a in analyses if a.get("type") == analysis_type]
            
        # Filter by date
        if date_from:
            try:
                date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                analyses = [a for a in analyses if datetime.strptime(a.get("timestamp").split()[0], "%Y-%m-%d") >= date_from_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_from: {date_from}")
                
        if date_to:
            try:
                date_to_dt = datetime.strptime(date_to, "%Y-%m-%d")
                analyses = [a for a in analyses if datetime.strptime(a.get("timestamp").split()[0], "%Y-%m-%d") <= date_to_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_to: {date_to}")
                
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit:
            analyses = analyses[:limit]
            
        return analyses
        
    def get_notes(self, limit=None, tags=None, date_from=None, date_to=None):
        """
        Get notes from the journal
        
        Args:
            limit (int, optional): Maximum number of notes to return
            tags (list, optional): Filter by tags
            date_from (str, optional): Filter by date (format: YYYY-MM-DD)
            date_to (str, optional): Filter by date (format: YYYY-MM-DD)
            
        Returns:
            list: List of notes
        """
        # Make a copy of notes
        notes = self.notes.copy()
        
        # Filter by tags
        if tags:
            notes = [n for n in notes if any(tag in n.get("tags", []) for tag in tags)]
            
        # Filter by date
        if date_from:
            try:
                date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                notes = [n for n in notes if datetime.strptime(n.get("timestamp").split()[0], "%Y-%m-%d") >= date_from_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_from: {date_from}")
                
        if date_to:
            try:
                date_to_dt = datetime.strptime(date_to, "%Y-%m-%d")
                notes = [n for n in notes if datetime.strptime(n.get("timestamp").split()[0], "%Y-%m-%d") <= date_to_dt]
            except ValueError:
                logger.error(f"Invalid date format for date_to: {date_to}")
                
        # Sort by timestamp (newest first)
        notes.sort(key=lambda n: n.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit:
            notes = notes[:limit]
            
        return notes
        
    def get_trade_statistics(self, date_from=None, date_to=None, symbol=None):
        """
        Get trade statistics
        
        Args:
            date_from (str, optional): Filter by date (format: YYYY-MM-DD)
            date_to (str, optional): Filter by date (format: YYYY-MM-DD)
            symbol (str, optional): Filter by symbol
            
        Returns:
            dict: Trade statistics
        """
        # Get filtered trades
        trades = self.get_trades(symbol=symbol, date_from=date_from, date_to=date_to)
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "average_pnl": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "average_win": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "average_hold_time": "0:00:00"
            }
            
        # Calculate statistics
        total_trades = len(trades)
        
        # Filter completed trades with profit/loss information
        completed_trades = [t for t in trades if "pnl" in t and t.get("status") == "Closed"]
        
        if not completed_trades:
            return {
                "total_trades": total_trades,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "average_pnl": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "average_win": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "average_hold_time": "0:00:00"
            }
            
        # Winning and losing trades
        winning_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in completed_trades if t.get("pnl", 0) < 0]
        
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        win_rate = (num_winning / len(completed_trades)) * 100 if completed_trades else 0
        
        # P&L statistics
        total_pnl = sum([t.get("pnl", 0) for t in completed_trades])
        average_pnl = total_pnl / len(completed_trades) if completed_trades else 0
        
        largest_win = max([t.get("pnl", 0) for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0
        
        average_win = sum([t.get("pnl", 0) for t in winning_trades]) / num_winning if num_winning > 0 else 0
        average_loss = sum([t.get("pnl", 0) for t in losing_trades]) / num_losing if num_losing > 0 else 0
        
        # Profit factor
        gross_profit = sum([t.get("pnl", 0) for t in winning_trades])
        gross_loss = abs(sum([t.get("pnl", 0) for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Hold time
        hold_times = []
        for t in completed_trades:
            if "entry_time" in t and "exit_time" in t:
                try:
                    entry_time = datetime.strptime(t["entry_time"], "%Y-%m-%d %H:%M:%S")
                    exit_time = datetime.strptime(t["exit_time"], "%Y-%m-%d %H:%M:%S")
                    hold_time = exit_time - entry_time
                    hold_times.append(hold_time.total_seconds())
                except ValueError:
                    pass
                    
        average_hold_seconds = sum(hold_times) / len(hold_times) if hold_times else 0
        average_hold_time = str(timedelta(seconds=int(average_hold_seconds)))
        
        return {
            "total_trades": total_trades,
            "winning_trades": num_winning,
            "losing_trades": num_losing,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_pnl": average_pnl,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "average_win": average_win,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "average_hold_time": average_hold_time
        }
        
    def get_strategy_statistics(self, date_from=None, date_to=None):
        """
        Get strategy statistics
        
        Args:
            date_from (str, optional): Filter by date (format: YYYY-MM-DD)
            date_to (str, optional): Filter by date (format: YYYY-MM-DD)
            
        Returns:
            dict: Strategy statistics
        """
        # Get filtered strategies
        strategies = self.get_strategies(date_from=date_from, date_to=date_to)
        
        if not strategies:
            return {
                "total_strategies": 0,
                "strategies_by_symbol": {},
                "strategies_by_type": {}
            }
            
        # Count by symbol
        symbol_counts = {}
        for s in strategies:
            symbol = s.get("symbol", "Unknown")
            if symbol not in symbol_counts:
                symbol_counts[symbol] = 0
            symbol_counts[symbol] += 1
            
        # Count by type
        type_counts = {}
        for s in strategies:
            s_type = s.get("type", "Unknown")
            if s_type not in type_counts:
                type_counts[s_type] = 0
            type_counts[s_type] += 1
            
        return {
            "total_strategies": len(strategies),
            "strategies_by_symbol": symbol_counts,
            "strategies_by_type": type_counts
        }
        
    def generate_trade_report(self, date_from=None, date_to=None, symbol=None):
        """
        Generate a comprehensive trade report
        
        Args:
            date_from (str, optional): Filter by date (format: YYYY-MM-DD)
            date_to (str, optional): Filter by date (format: YYYY-MM-DD)
            symbol (str, optional): Filter by symbol
            
        Returns:
            dict: Report data
        """
        # Get filtered trades
        trades = self.get_trades(symbol=symbol, date_from=date_from, date_to=date_to)
        
        # Get statistics
        stats = self.get_trade_statistics(date_from=date_from, date_to=date_to, symbol=symbol)
        
        # Get trade data for charts
        dates = []
        pnls = []
        cumulative_pnl = []
        
        for t in trades:
            if "timestamp" in t and "pnl" in t:
                dates.append(t["timestamp"].split()[0])
                pnls.append(t.get("pnl", 0))
                
        # Sort by date
        date_pnl_pairs = sorted(zip(dates, pnls), key=lambda x: x[0])
        dates = [d for d, _ in date_pnl_pairs]
        pnls = [p for _, p in date_pnl_pairs]
        
        # Calculate cumulative P&L
        cum_pnl = 0
        for pnl in pnls:
            cum_pnl += pnl
            cumulative_pnl.append(cum_pnl)
            
        # Group trades by day
        daily_pnl = {}
        for t in trades:
            if "timestamp" in t and "pnl" in t:
                date = t["timestamp"].split()[0]
                if date not in daily_pnl:
                    daily_pnl[date] = 0
                daily_pnl[date] += t.get("pnl", 0)
                
        # Get daily pnl
        daily_dates = sorted(daily_pnl.keys())
        daily_pnls = [daily_pnl[d] for d in daily_dates]
        
        # Categorize trades
        trades_by_symbol = {}
        for t in trades:
            symbol = t.get("symbol", "Unknown")
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(t)
            
        # Generate symbol statistics
        symbol_stats = {}
        for sym, sym_trades in trades_by_symbol.items():
            completed_trades = [t for t in sym_trades if "pnl" in t and t.get("status") == "Closed"]
            
            if not completed_trades:
                continue
                
            winning_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]
            total_pnl = sum([t.get("pnl", 0) for t in completed_trades])
            win_rate = (len(winning_trades) / len(completed_trades)) * 100
            
            symbol_stats[sym] = {
                "count": len(sym_trades),
                "total_pnl": total_pnl,
                "win_rate": win_rate
            }
            
        # Return complete report
        return {
            "summary": stats,
            "trades": trades,
            "trades_by_symbol": trades_by_symbol,
            "symbol_stats": symbol_stats,
            "chart_data": {
                "dates": dates,
                "pnls": pnls,
                "cumulative_pnl": cumulative_pnl,
                "daily_dates": daily_dates,
                "daily_pnls": daily_pnls
            }
        }
        
    def export_to_csv(self, file_path):
        """
        Export journal data to CSV files
        
        Args:
            file_path (str): Directory path to save files
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(file_path, exist_ok=True)
            
            # Export trades
            if self.trade_entries:
                trades_df = pd.DataFrame(self.trade_entries)
                trades_df.to_csv(os.path.join(file_path, "trades.csv"), index=False)
                
            # Export strategies
            if self.strategy_entries:
                strategies_df = pd.DataFrame(self.strategy_entries)
                strategies_df.to_csv(os.path.join(file_path, "strategies.csv"), index=False)
                
            # Export market analyses
            if self.market_entries:
                market_df = pd.DataFrame(self.market_entries)
                market_df.to_csv(os.path.join(file_path, "market_analyses.csv"), index=False)
                
            # Export notes
            if self.notes:
                notes_df = pd.DataFrame(self.notes)
                notes_df.to_csv(os.path.join(file_path, "notes.csv"), index=False)
                
            logger.info(f"Exported journal data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting journal data: {str(e)}")
            return False
            
    def _save_data(self):
        """
        Save journal data to disk
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            # Prepare data to save
            data = {
                "trade_entries": self.trade_entries,
                "strategy_entries": self.strategy_entries,
                "market_entries": self.market_entries,
                "notes": self.notes,
                "max_entries": self.max_entries,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open("config/trading_journal.json", "w") as f:
                json.dump(data, f, indent=4)
                
            logger.info("Saved trading journal data")
        except Exception as e:
            logger.error(f"Error saving trading journal data: {str(e)}")
            
    def _load_data(self):
        """
        Load journal data from disk
        """
        try:
            # Check if data file exists
            if not os.path.exists("config/trading_journal.json"):
                logger.info("No trading journal data file found, using defaults")
                return
                
            # Load data
            with open("config/trading_journal.json", "r") as f:
                data = json.load(f)
                
            # Apply settings
            if "trade_entries" in data:
                self.trade_entries = data["trade_entries"]
                
            if "strategy_entries" in data:
                self.strategy_entries = data["strategy_entries"]
                
            if "market_entries" in data:
                self.market_entries = data["market_entries"]
                
            if "notes" in data:
                self.notes = data["notes"]
                
            if "max_entries" in data:
                self.max_entries = data["max_entries"]
                
            logger.info("Loaded trading journal data")
        except Exception as e:
            logger.error(f"Error loading trading journal data: {str(e)}")