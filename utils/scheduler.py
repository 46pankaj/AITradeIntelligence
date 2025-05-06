import logging
import time
import threading
from datetime import datetime, time as dt_time, timedelta
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingScheduler:
    """
    Class for scheduling automated trading based on market hours and days
    """
    def __init__(self):
        # Default trading hours for NSE (9:15 AM to 3:30 PM IST)
        self.trading_start_time = dt_time(9, 15)
        self.trading_end_time = dt_time(15, 30)
        
        # Default pre-market time (30 min before market open)
        self.pre_market_minutes = 30
        
        # Default trading days (Monday to Friday)
        self.trading_days = [0, 1, 2, 3, 4]  # Monday=0, Sunday=6
        
        # Indian timezone for market hours
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        # For holiday calendar
        self.holidays = []
        
        # Thread control
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.pre_market_callback = None
        self.market_open_callback = None
        self.market_close_callback = None
        self.trading_cycle_callback = None
        
        # Trading cycle interval (in minutes)
        self.trading_cycle_interval = 15
        
    def set_trading_hours(self, start_hour, start_minute, end_hour, end_minute):
        """
        Set trading hours
        
        Args:
            start_hour (int): Hour to start trading (24h format)
            start_minute (int): Minute to start trading
            end_hour (int): Hour to end trading (24h format)
            end_minute (int): Minute to end trading
        """
        self.trading_start_time = dt_time(start_hour, start_minute)
        self.trading_end_time = dt_time(end_hour, end_minute)
        logger.info(f"Trading hours set to {start_hour}:{start_minute} - {end_hour}:{end_minute}")
        
    def set_pre_market_time(self, minutes_before):
        """
        Set pre-market time in minutes before market open
        
        Args:
            minutes_before (int): Minutes before market open to start pre-market activities
        """
        self.pre_market_minutes = minutes_before
        logger.info(f"Pre-market time set to {minutes_before} minutes before market open")
        
    def set_trading_days(self, days):
        """
        Set trading days (0=Monday, 6=Sunday)
        
        Args:
            days (list): List of days to trade
        """
        self.trading_days = days
        days_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
        days_str = ", ".join([days_names[day] for day in days])
        logger.info(f"Trading days set to {days_str}")
        
    def add_holiday(self, date_str):
        """
        Add a holiday date (no trading on this date)
        
        Args:
            date_str (str): Date string in format 'YYYY-MM-DD'
        """
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            self.holidays.append(date_obj)
            logger.info(f"Added holiday: {date_str}")
        except ValueError:
            logger.error(f"Invalid date format for holiday: {date_str}. Use 'YYYY-MM-DD'.")
            
    def clear_holidays(self):
        """Clear all holidays"""
        self.holidays = []
        logger.info("Cleared all holidays")
        
    def set_callbacks(self, pre_market=None, market_open=None, market_close=None, trading_cycle=None):
        """
        Set callback functions for different market events
        
        Args:
            pre_market (callable): Called during pre-market time
            market_open (callable): Called when market opens
            market_close (callable): Called when market closes
            trading_cycle (callable): Called at regular intervals during trading hours
        """
        if pre_market:
            self.pre_market_callback = pre_market
            
        if market_open:
            self.market_open_callback = market_open
            
        if market_close:
            self.market_close_callback = market_close
            
        if trading_cycle:
            self.trading_cycle_callback = trading_cycle
            
    def set_trading_cycle_interval(self, minutes):
        """
        Set interval for trading cycle callbacks in minutes
        
        Args:
            minutes (int): Interval in minutes
        """
        self.trading_cycle_interval = minutes
        logger.info(f"Trading cycle interval set to {minutes} minutes")
        
    def is_trading_day(self, date=None):
        """
        Check if the given date is a trading day
        
        Args:
            date (datetime.date, optional): Date to check. Defaults to today.
            
        Returns:
            bool: True if trading day, False otherwise
        """
        if date is None:
            date = datetime.now(self.timezone).date()
            
        # Check if it's a configured trading day (by weekday)
        if date.weekday() not in self.trading_days:
            return False
            
        # Check if it's a holiday
        if date in self.holidays:
            return False
            
        return True
        
    def is_trading_hours(self, time=None):
        """
        Check if the given time is during trading hours
        
        Args:
            time (datetime.time, optional): Time to check. Defaults to now.
            
        Returns:
            bool: True if during trading hours, False otherwise
        """
        if time is None:
            time = datetime.now(self.timezone).time()
            
        return self.trading_start_time <= time <= self.trading_end_time
        
    def is_pre_market(self, time=None):
        """
        Check if the given time is during pre-market hours
        
        Args:
            time (datetime.time, optional): Time to check. Defaults to now.
            
        Returns:
            bool: True if during pre-market hours, False otherwise
        """
        if time is None:
            time = datetime.now(self.timezone).time()
            
        # Calculate pre-market start time
        pre_market_start_dt = datetime.combine(datetime.today(), self.trading_start_time) - timedelta(minutes=self.pre_market_minutes)
        pre_market_start = pre_market_start_dt.time()
        
        return pre_market_start <= time < self.trading_start_time
        
    def time_to_next_market_open(self):
        """
        Calculate time to next market open
        
        Returns:
            float: Seconds until next market open
        """
        now = datetime.now(self.timezone)
        today = now.date()
        
        # Create datetime for today's market open
        market_open_dt = datetime.combine(today, self.trading_start_time)
        market_open_dt = self.timezone.localize(market_open_dt)
        
        # If market open is in the past, look for next trading day
        if now >= market_open_dt:
            days_ahead = 1
            while days_ahead < 8:  # Look up to a week ahead
                next_date = today + timedelta(days=days_ahead)
                if self.is_trading_day(next_date):
                    market_open_dt = datetime.combine(next_date, self.trading_start_time)
                    market_open_dt = self.timezone.localize(market_open_dt)
                    break
                days_ahead += 1
                
        time_diff = market_open_dt - now
        return max(0, time_diff.total_seconds())
        
    def start_scheduler(self):
        """
        Start the scheduler thread
        
        Returns:
            bool: Success status
        """
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler is already running")
            return False
            
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Started trading scheduler")
        return True
        
    def stop_scheduler(self):
        """
        Stop the scheduler thread
        
        Returns:
            bool: Success status
        """
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("Scheduler is not running")
            return False
            
        self.stop_event.set()
        self.scheduler_thread.join(timeout=10)
        
        logger.info("Stopped trading scheduler")
        return True
        
    def _scheduler_loop(self):
        """
        Main scheduler loop
        """
        logger.info("Starting scheduler loop")
        
        # Track state
        in_pre_market = False
        in_trading_hours = False
        last_cycle_time = None
        
        while not self.stop_event.is_set():
            try:
                now = datetime.now(self.timezone)
                current_time = now.time()
                
                # Check if today is a trading day
                if not self.is_trading_day(now.date()):
                    # Sleep until midnight and check again
                    seconds_to_midnight = (datetime.combine(now.date() + timedelta(days=1), dt_time(0, 0)) - now).total_seconds()
                    logger.info(f"Not a trading day. Sleeping until midnight ({seconds_to_midnight:.0f} seconds)")
                    
                    # Sleep in smaller increments to check for stop event
                    for _ in range(int(seconds_to_midnight / 60)):
                        if self.stop_event.is_set():
                            break
                        time.sleep(60)
                    continue
                
                # Check if we're in pre-market
                is_pre_market_now = self.is_pre_market(current_time)
                if is_pre_market_now and not in_pre_market:
                    # Just entered pre-market
                    logger.info("Entering pre-market time")
                    in_pre_market = True
                    
                    if self.pre_market_callback:
                        try:
                            self.pre_market_callback()
                        except Exception as e:
                            logger.error(f"Error in pre-market callback: {str(e)}")
                            
                elif not is_pre_market_now and in_pre_market:
                    # Just exited pre-market
                    in_pre_market = False
                
                # Check if we're in trading hours
                is_trading_hours_now = self.is_trading_hours(current_time)
                if is_trading_hours_now and not in_trading_hours:
                    # Just entered trading hours
                    logger.info("Market is now open")
                    in_trading_hours = True
                    
                    if self.market_open_callback:
                        try:
                            self.market_open_callback()
                        except Exception as e:
                            logger.error(f"Error in market open callback: {str(e)}")
                            
                    # Initialize last cycle time
                    last_cycle_time = now
                    
                elif not is_trading_hours_now and in_trading_hours:
                    # Just exited trading hours
                    logger.info("Market is now closed")
                    in_trading_hours = False
                    
                    if self.market_close_callback:
                        try:
                            self.market_close_callback()
                        except Exception as e:
                            logger.error(f"Error in market close callback: {str(e)}")
                
                # Check if we need to run a trading cycle
                if in_trading_hours and self.trading_cycle_callback:
                    # Check if it's time for a new trading cycle
                    time_diff = now - last_cycle_time
                    if time_diff.total_seconds() >= self.trading_cycle_interval * 60:
                        logger.info(f"Running trading cycle ({self.trading_cycle_interval} minute interval)")
                        try:
                            self.trading_cycle_callback()
                        except Exception as e:
                            logger.error(f"Error in trading cycle callback: {str(e)}")
                            
                        last_cycle_time = now
                
                # Sleep before checking again (1 minute)
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)  # Sleep on error
                
    def get_schedule(self):
        """
        Get the current trading schedule
        
        Returns:
            dict: Trading schedule
        """
        # Initialize schedule with existing values
        if not hasattr(self, 'trading_schedule'):
            self.trading_schedule = {
                'monday': {'enabled': 0 in self.trading_days, 'start_time': self.trading_start_time.strftime('%H:%M'), 'end_time': self.trading_end_time.strftime('%H:%M')},
                'tuesday': {'enabled': 1 in self.trading_days, 'start_time': self.trading_start_time.strftime('%H:%M'), 'end_time': self.trading_end_time.strftime('%H:%M')},
                'wednesday': {'enabled': 2 in self.trading_days, 'start_time': self.trading_start_time.strftime('%H:%M'), 'end_time': self.trading_end_time.strftime('%H:%M')},
                'thursday': {'enabled': 3 in self.trading_days, 'start_time': self.trading_start_time.strftime('%H:%M'), 'end_time': self.trading_end_time.strftime('%H:%M')},
                'friday': {'enabled': 4 in self.trading_days, 'start_time': self.trading_start_time.strftime('%H:%M'), 'end_time': self.trading_end_time.strftime('%H:%M')}
            }
            
            # Add strategies list if not present
            if not hasattr(self, 'strategy_ids'):
                self.strategy_ids = []
            self.trading_schedule['strategies'] = self.strategy_ids
            
        return self.trading_schedule
        
    def update_schedule(self, day, enabled, start_time, end_time):
        """
        Update the trading schedule for a specific day
        
        Args:
            day (str): Day of the week (e.g., 'monday')
            enabled (bool): Whether trading is enabled on this day
            start_time (str): Start time in format 'HH:MM'
            end_time (str): End time in format 'HH:MM'
            
        Returns:
            bool: Success status
        """
        if not hasattr(self, 'trading_schedule'):
            self.get_schedule()  # Initialize schedule
            
        if day not in self.trading_schedule:
            logger.error(f"Invalid day for schedule update: {day}")
            return False
            
        # Update schedule
        self.trading_schedule[day] = {
            'enabled': enabled,
            'start_time': start_time,
            'end_time': end_time
        }
        
        # Update trading days based on enabled days
        day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
        self.trading_days = [day_map[d] for d in self.trading_schedule if d != 'strategies' and self.trading_schedule[d].get('enabled', False)]
        
        # Update trading hours based on the first enabled day's schedule
        for d, settings in self.trading_schedule.items():
            if d != 'strategies' and settings.get('enabled', False):
                try:
                    start_hour, start_minute = map(int, settings['start_time'].split(':'))
                    end_hour, end_minute = map(int, settings['end_time'].split(':'))
                    self.set_trading_hours(start_hour, start_minute, end_hour, end_minute)
                except (ValueError, KeyError) as e:
                    logger.error(f"Error setting trading hours from schedule: {str(e)}")
                break
                
        logger.info(f"Updated trading schedule for {day}")
        return True
        
    def get_credentials(self):
        """
        Get stored API credentials
        
        Returns:
            dict: API credentials or None if not set
        """
        if not hasattr(self, 'api_credentials'):
            return None
        return self.api_credentials
        
    def store_credentials(self, username, password, totp_secret=None):
        """
        Store API credentials for automatic login
        
        Args:
            username (str): API username
            password (str): API password
            totp_secret (str, optional): TOTP secret for 2FA
            
        Returns:
            bool: Success status
        """
        try:
            self.api_credentials = {
                'username': username,
                'password': password
            }
            
            if totp_secret:
                self.api_credentials['totp_secret'] = totp_secret
                
            logger.info(f"Stored API credentials for {username}")
            return True
        except Exception as e:
            logger.error(f"Error storing API credentials: {str(e)}")
            return False
            
    def add_strategy_to_schedule(self, strategy_id):
        """
        Add a strategy to the automated trading schedule
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            bool: Success status
        """
        if not hasattr(self, 'strategy_ids'):
            self.strategy_ids = []
            
        if strategy_id not in self.strategy_ids:
            self.strategy_ids.append(strategy_id)
            
            # Update in trading schedule
            if hasattr(self, 'trading_schedule'):
                self.trading_schedule['strategies'] = self.strategy_ids
                
            logger.info(f"Added strategy {strategy_id} to trading schedule")
            return True
        return False
        
    def remove_strategy_from_schedule(self, strategy_id):
        """
        Remove a strategy from the automated trading schedule
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            bool: Success status
        """
        if not hasattr(self, 'strategy_ids'):
            return False
            
        if strategy_id in self.strategy_ids:
            self.strategy_ids.remove(strategy_id)
            
            # Update in trading schedule
            if hasattr(self, 'trading_schedule'):
                self.trading_schedule['strategies'] = self.strategy_ids
                
            logger.info(f"Removed strategy {strategy_id} from trading schedule")
            return True
        return False