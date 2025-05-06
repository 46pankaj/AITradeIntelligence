import logging
import json
import os
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionManager:
    """
    Class for managing trading positions, capital allocation, and risk management
    """
    def __init__(self, initial_capital=1000000):
        # Trading capital
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.allocated_capital = 0
        
        # Risk management
        self.max_risk_per_trade = 2.0  # As percentage of available capital
        self.max_risk_per_symbol = 10.0  # As percentage of available capital
        self.max_portfolio_risk = 25.0  # As percentage of total capital
        
        # Position sizing
        self.position_size_mode = "volatility"  # Options: fixed, volatility, signal_strength
        self.default_position_size = 25000  # Default size for fixed mode
        
        # Diversification
        self.max_positions = 5  # Maximum number of concurrent positions
        self.max_positions_per_sector = 2  # Maximum positions per sector
        self.max_exposure_per_sector = 25.0  # As percentage of allocated capital
        self.sector_allocations = {}  # Target allocation per sector
        
        # Position tracking
        self.positions = []
        self.position_history = []
        
        # Volatility multipliers for position sizing
        self.volatility_multipliers = {
            "Very Low": 1.5,  # Increase position size for low volatility
            "Low": 1.2,
            "Medium": 1.0,
            "High": 0.8,
            "Very High": 0.6  # Decrease position size for high volatility
        }
        
        # Signal strength multipliers for position sizing
        self.signal_multipliers = {
            "Very Weak": 0.5,
            "Weak": 0.8,
            "Moderate": 1.0,
            "Strong": 1.2,
            "Very Strong": 1.5
        }
        
        # Load saved positions and settings
        self._load_data()
        
    def set_capital(self, amount):
        """
        Set available trading capital
        
        Args:
            amount (float): Capital amount
        """
        self.initial_capital = amount
        # Only update available capital if no positions are open
        if not self.positions:
            self.available_capital = amount
            
        logger.info(f"Set trading capital to ₹{amount:,.2f}")
        self._save_data()
        
    def set_risk_parameters(self, max_risk_per_trade, max_risk_per_symbol, max_portfolio_risk):
        """
        Set risk management parameters
        
        Args:
            max_risk_per_trade (float): Maximum risk per trade as percentage of available capital
            max_risk_per_symbol (float): Maximum risk per symbol as percentage of available capital
            max_portfolio_risk (float): Maximum portfolio risk as percentage of total capital
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_per_symbol = max_risk_per_symbol
        self.max_portfolio_risk = max_portfolio_risk
        
        logger.info(f"Set risk parameters: Max Risk Per Trade={max_risk_per_trade}%, Max Risk Per Symbol={max_risk_per_symbol}%, Max Portfolio Risk={max_portfolio_risk}%")
        self._save_data()
        
    def set_position_sizing(self, mode, default_size=None):
        """
        Set position sizing mode
        
        Args:
            mode (str): Position sizing mode (fixed, volatility, signal_strength)
            default_size (float, optional): Default position size for fixed mode
        """
        valid_modes = ["fixed", "volatility", "signal_strength"]
        if mode not in valid_modes:
            logger.error(f"Invalid position sizing mode: {mode}. Must be one of {valid_modes}")
            return
            
        self.position_size_mode = mode
        
        if default_size:
            self.default_position_size = default_size
            
        logger.info(f"Set position sizing mode to {mode}" + (f" with default size ₹{default_size:,.2f}" if default_size else ""))
        self._save_data()
        
    def set_diversification_limits(self, max_positions, max_positions_per_sector, max_exposure_per_sector):
        """
        Set diversification limits
        
        Args:
            max_positions (int): Maximum number of concurrent positions
            max_positions_per_sector (int): Maximum positions per sector
            max_exposure_per_sector (float): Maximum exposure per sector as percentage of allocated capital
        """
        self.max_positions = max_positions
        self.max_positions_per_sector = max_positions_per_sector
        self.max_exposure_per_sector = max_exposure_per_sector
        
        logger.info(f"Set diversification limits: Max Positions={max_positions}, Max Positions Per Sector={max_positions_per_sector}, Max Exposure Per Sector={max_exposure_per_sector}%")
        self._save_data()
        
    def set_sector_allocations(self, allocations):
        """
        Set target sector allocations
        
        Args:
            allocations (dict): Dictionary of sector allocations (e.g., {"Technology": 25.0, "Finance": 20.0})
        """
        # Validate that allocations sum to 100% or less
        total_allocation = sum(allocations.values())
        if total_allocation > 100:
            logger.error(f"Total sector allocation ({total_allocation}%) exceeds 100%")
            return
            
        self.sector_allocations = allocations
        
        logger.info(f"Set sector allocations: {allocations}")
        self._save_data()
        
    def calculate_position_size(self, symbol, price, stop_loss_percent, volatility=None, signal_strength=None, sector=None):
        """
        Calculate appropriate position size based on risk parameters and current market conditions
        
        Args:
            symbol (str): Trading symbol
            price (float): Current price
            stop_loss_percent (float): Stop loss percentage
            volatility (str or float, optional): Volatility level (Very Low, Low, Medium, High, Very High) or actual value
            signal_strength (str or float, optional): Signal strength (Very Weak, Weak, Moderate, Strong, Very Strong) or actual value
            sector (str, optional): Market sector for the symbol
            
        Returns:
            dict: Position size details
        """
        # Check available capital
        if self.available_capital <= 0:
            logger.warning("No available capital for new positions")
            return {
                "size": 0,
                "quantity": 0,
                "message": "No available capital available"
            }
            
        # Check total number of positions
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum number of positions ({self.max_positions}) already reached")
            return {
                "size": 0,
                "quantity": 0,
                "message": f"Maximum positions ({self.max_positions}) reached"
            }
            
        # Check positions per sector
        if sector:
            sector_positions = [p for p in self.positions if p.get("sector") == sector]
            if len(sector_positions) >= self.max_positions_per_sector:
                logger.warning(f"Maximum positions for sector {sector} ({self.max_positions_per_sector}) already reached")
                return {
                    "size": 0,
                    "quantity": 0,
                    "message": f"Maximum positions for sector {sector} reached"
                }
                
        # Calculate risk amount based on max_risk_per_trade
        risk_amount = self.available_capital * (self.max_risk_per_trade / 100)
        
        # Calculate sector exposure if applicable
        if sector:
            current_sector_allocation = self._get_current_sector_allocation(sector)
            target_allocation = self.sector_allocations.get(sector, 0)
            
            if target_allocation > 0:
                # Adjust risk amount based on target allocation
                allocation_factor = target_allocation / current_sector_allocation if current_sector_allocation > 0 else 2.0
                allocation_factor = min(2.0, max(0.5, allocation_factor))  # Limit adjustment factor
                risk_amount *= allocation_factor
                
        # Calculate position size based on stop loss
        if stop_loss_percent <= 0:
            logger.warning("Invalid stop loss percentage (must be > 0)")
            stop_loss_percent = 5.0  # Default to 5%
            
        # Base position size calculation (risk amount / stop loss percentage)
        position_size = risk_amount / (stop_loss_percent / 100)
        
        # Apply position sizing mode adjustments
        if self.position_size_mode == "fixed":
            position_size = min(self.default_position_size, self.available_capital)
        elif self.position_size_mode == "volatility" and volatility:
            # Apply volatility multiplier
            if isinstance(volatility, str):
                # Use predefined category
                multiplier = self.volatility_multipliers.get(volatility, 1.0)
            else:
                # Map numeric volatility to a multiplier (assuming volatility is in percentage)
                if volatility < 0.5:
                    multiplier = self.volatility_multipliers["Very Low"]
                elif volatility < 1.0:
                    multiplier = self.volatility_multipliers["Low"]
                elif volatility < 2.0:
                    multiplier = self.volatility_multipliers["Medium"]
                elif volatility < 3.0:
                    multiplier = self.volatility_multipliers["High"]
                else:
                    multiplier = self.volatility_multipliers["Very High"]
                    
            position_size *= multiplier
            
        elif self.position_size_mode == "signal_strength" and signal_strength:
            # Apply signal strength multiplier
            if isinstance(signal_strength, str):
                # Use predefined category
                multiplier = self.signal_multipliers.get(signal_strength, 1.0)
            else:
                # Map numeric signal strength (0-1) to a multiplier
                if signal_strength < 0.3:
                    multiplier = self.signal_multipliers["Very Weak"]
                elif signal_strength < 0.5:
                    multiplier = self.signal_multipliers["Weak"]
                elif signal_strength < 0.7:
                    multiplier = self.signal_multipliers["Moderate"]
                elif signal_strength < 0.9:
                    multiplier = self.signal_multipliers["Strong"]
                else:
                    multiplier = self.signal_multipliers["Very Strong"]
                    
            position_size *= multiplier
            
        # Ensure position size doesn't exceed available capital
        position_size = min(position_size, self.available_capital)
        
        # Calculate quantity based on price
        quantity = int(position_size / price)
        actual_size = quantity * price
        
        # Add some contextual information for the caller
        return {
            "size": actual_size,
            "quantity": quantity,
            "message": f"Position size calculated based on {self.position_size_mode} strategy"
        }
        
    def add_position(self, position_data):
        """
        Add a new trading position
        
        Args:
            position_data (dict): Position data including symbol, entry_price, quantity, etc.
            
        Returns:
            str: Position ID
        """
        # Check required fields
        required_fields = ["symbol", "entry_price", "quantity", "type"]
        for field in required_fields:
            if field not in position_data:
                logger.error(f"Missing required field: {field}")
                return None
                
        symbol = position_data["symbol"]
        entry_price = position_data["entry_price"]
        quantity = position_data["quantity"]
        
        # Calculate position value
        position_value = entry_price * quantity
        
        # Check if enough capital is available
        if position_value > self.available_capital:
            logger.error(f"Insufficient capital for position: ₹{position_value:,.2f} required, ₹{self.available_capital:,.2f} available")
            return None
            
        # Generate a unique position ID
        position_id = f"POS-{len(self.positions)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add additional fields
        position_data["id"] = position_id
        position_data["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        position_data["value"] = position_value
        position_data["status"] = "Open"
        
        # Calculate current P&L (initially zero)
        position_data["current_price"] = entry_price
        position_data["current_value"] = position_value
        position_data["pnl"] = 0
        position_data["pnl_percent"] = 0
        
        # Add position
        self.positions.append(position_data)
        
        # Update available capital
        self.available_capital -= position_value
        self.allocated_capital += position_value
        
        logger.info(f"Added position: {symbol}, {position_data['type']} {quantity} @ ₹{entry_price:,.2f} (₹{position_value:,.2f})")
        self._save_data()
        
        return position_id
        
    def update_position(self, position_id, current_price=None, update_data=None):
        """
        Update an existing position
        
        Args:
            position_id (str): Position ID
            current_price (float, optional): Current price for the position
            update_data (dict, optional): Additional data to update
            
        Returns:
            bool: Success status
        """
        # Find position
        position = None
        for p in self.positions:
            if p["id"] == position_id:
                position = p
                break
                
        if not position:
            logger.error(f"Position not found: {position_id}")
            return False
            
        # Update price and P&L if provided
        if current_price:
            position["current_price"] = current_price
            position["current_value"] = current_price * position["quantity"]
            
            # Calculate P&L
            if position["type"] == "BUY":
                pnl = position["current_value"] - position["value"]
            else:  # SELL
                pnl = position["value"] - position["current_value"]
                
            position["pnl"] = pnl
            position["pnl_percent"] = (pnl / position["value"]) * 100
            
        # Update additional data
        if update_data:
            for key, value in update_data.items():
                # Don't update certain protected fields
                if key not in ["id", "entry_time", "value", "type"]:
                    position[key] = value
                    
        logger.info(f"Updated position: {position_id}")
        self._save_data()
        
        return True
        
    def close_position(self, position_id, exit_price, exit_reason="Manual"):
        """
        Close an existing position
        
        Args:
            position_id (str): Position ID
            exit_price (float): Exit price
            exit_reason (str, optional): Reason for closing the position
            
        Returns:
            dict: Closed position details
        """
        # Find position
        position_index = None
        position = None
        for i, p in enumerate(self.positions):
            if p["id"] == position_id:
                position_index = i
                position = p
                break
                
        if position_index is None:
            logger.error(f"Position not found: {position_id}")
            return None
            
        # Calculate final P&L
        if position["type"] == "BUY":
            pnl = (exit_price - position["entry_price"]) * position["quantity"]
        else:  # SELL
            pnl = (position["entry_price"] - exit_price) * position["quantity"]
            
        pnl_percent = (pnl / position["value"]) * 100
        
        # Update position data
        position["exit_price"] = exit_price
        position["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        position["pnl"] = pnl
        position["pnl_percent"] = pnl_percent
        position["exit_reason"] = exit_reason
        position["status"] = "Closed"
        
        # Return capital to available capital
        self.available_capital += position["value"] + pnl
        self.allocated_capital -= position["value"]
        
        # Add to history and remove from active positions
        self.position_history.append(position)
        self.positions.pop(position_index)
        
        logger.info(f"Closed position: {position['symbol']}, {position['type']} {position['quantity']} @ ₹{exit_price:,.2f}, P&L: ₹{pnl:,.2f} ({pnl_percent:.2f}%)")
        self._save_data()
        
        return position
        
    def get_position(self, position_id):
        """
        Get details for a specific position
        
        Args:
            position_id (str): Position ID
            
        Returns:
            dict: Position details or None if not found
        """
        for p in self.positions:
            if p["id"] == position_id:
                return p
                
        # Check history as well
        for p in self.position_history:
            if p["id"] == position_id:
                return p
                
        return None
        
    def get_positions_by_symbol(self, symbol):
        """
        Get all positions for a specific symbol
        
        Args:
            symbol (str): Symbol to filter by
            
        Returns:
            list: List of positions for the symbol
        """
        return [p for p in self.positions if p["symbol"] == symbol]
        
    def get_positions_by_sector(self, sector):
        """
        Get all positions for a specific sector
        
        Args:
            sector (str): Sector to filter by
            
        Returns:
            list: List of positions for the sector
        """
        return [p for p in self.positions if p.get("sector") == sector]
        
    def get_sector_exposure(self, sector):
        """
        Get total exposure for a specific sector
        
        Args:
            sector (str): Sector to calculate exposure for
            
        Returns:
            float: Total exposure in the sector
        """
        return sum([p["value"] for p in self.positions if p.get("sector") == sector])
        
    def get_portfolio_stats(self):
        """
        Get current portfolio statistics
        
        Returns:
            dict: Portfolio statistics
        """
        # Calculate total position value and P&L
        total_value = sum([p["current_value"] for p in self.positions]) if self.positions else 0
        total_pnl = sum([p["pnl"] for p in self.positions]) if self.positions else 0
        
        # Calculate P&L percentage relative to allocated capital
        pnl_percent = (total_pnl / self.allocated_capital) * 100 if self.allocated_capital > 0 else 0
        
        # Calculate sector exposures
        sector_exposures = {}
        for p in self.positions:
            sector = p.get("sector", "Unknown")
            if sector not in sector_exposures:
                sector_exposures[sector] = 0
            sector_exposures[sector] += p["value"]
            
        # Calculate sector allocations (as percentage of allocated capital)
        sector_allocations = {}
        for sector, exposure in sector_exposures.items():
            sector_allocations[sector] = (exposure / self.allocated_capital) * 100 if self.allocated_capital > 0 else 0
            
        return {
            "total_capital": self.initial_capital,
            "available_capital": self.available_capital,
            "allocated_capital": self.allocated_capital,
            "total_positions": len(self.positions),
            "total_position_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_percent": pnl_percent,
            "sector_exposures": sector_exposures,
            "sector_allocations": sector_allocations
        }
        
    def reset(self):
        """
        Reset position manager (close all positions and reset capital)
        
        Returns:
            bool: Success status
        """
        # Close all positions and move to history
        for position in self.positions:
            position["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            position["exit_reason"] = "Position Manager Reset"
            position["status"] = "Closed"
            self.position_history.append(position)
            
        # Reset capital
        self.available_capital = self.initial_capital
        self.allocated_capital = 0
        self.positions = []
        
        logger.info("Reset position manager")
        self._save_data()
        
        return True
        
    def _get_current_sector_allocation(self, sector):
        """
        Get current allocation for a sector as percentage of allocated capital
        
        Args:
            sector (str): Sector to get allocation for
            
        Returns:
            float: Current allocation as percentage
        """
        sector_exposure = self.get_sector_exposure(sector)
        return (sector_exposure / self.allocated_capital) * 100 if self.allocated_capital > 0 else 0
        
    def _save_data(self):
        """
        Save position manager data to disk
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            # Prepare data to save
            data = {
                "capital": {
                    "initial": self.initial_capital,
                    "available": self.available_capital,
                    "allocated": self.allocated_capital
                },
                "risk": {
                    "max_risk_per_trade": self.max_risk_per_trade,
                    "max_risk_per_symbol": self.max_risk_per_symbol,
                    "max_portfolio_risk": self.max_portfolio_risk
                },
                "position_sizing": {
                    "mode": self.position_size_mode,
                    "default_size": self.default_position_size,
                    "volatility_multipliers": self.volatility_multipliers,
                    "signal_multipliers": self.signal_multipliers
                },
                "diversification": {
                    "max_positions": self.max_positions,
                    "max_positions_per_sector": self.max_positions_per_sector,
                    "max_exposure_per_sector": self.max_exposure_per_sector,
                    "sector_allocations": self.sector_allocations
                },
                "positions": self.positions,
                "position_history": self.position_history,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open("config/position_manager.json", "w") as f:
                json.dump(data, f, indent=4)
                
            logger.info("Saved position manager data")
        except Exception as e:
            logger.error(f"Error saving position manager data: {str(e)}")
            
    def _load_data(self):
        """
        Load position manager data from disk
        """
        try:
            # Check if data file exists
            if not os.path.exists("config/position_manager.json"):
                logger.info("No position manager data file found, using defaults")
                return
                
            # Load data
            with open("config/position_manager.json", "r") as f:
                data = json.load(f)
                
            # Apply settings
            if "capital" in data:
                self.initial_capital = data["capital"].get("initial", self.initial_capital)
                self.available_capital = data["capital"].get("available", self.available_capital)
                self.allocated_capital = data["capital"].get("allocated", self.allocated_capital)
                
            if "risk" in data:
                self.max_risk_per_trade = data["risk"].get("max_risk_per_trade", self.max_risk_per_trade)
                self.max_risk_per_symbol = data["risk"].get("max_risk_per_symbol", self.max_risk_per_symbol)
                self.max_portfolio_risk = data["risk"].get("max_portfolio_risk", self.max_portfolio_risk)
                
            if "position_sizing" in data:
                self.position_size_mode = data["position_sizing"].get("mode", self.position_size_mode)
                self.default_position_size = data["position_sizing"].get("default_size", self.default_position_size)
                self.volatility_multipliers.update(data["position_sizing"].get("volatility_multipliers", {}))
                self.signal_multipliers.update(data["position_sizing"].get("signal_multipliers", {}))
                
            if "diversification" in data:
                self.max_positions = data["diversification"].get("max_positions", self.max_positions)
                self.max_positions_per_sector = data["diversification"].get("max_positions_per_sector", self.max_positions_per_sector)
                self.max_exposure_per_sector = data["diversification"].get("max_exposure_per_sector", self.max_exposure_per_sector)
                self.sector_allocations = data["diversification"].get("sector_allocations", self.sector_allocations)
                
            if "positions" in data:
                self.positions = data["positions"]
                
            if "position_history" in data:
                self.position_history = data["position_history"]
                
            logger.info("Loaded position manager data")
        except Exception as e:
            logger.error(f"Error loading position manager data: {str(e)}")