import logging
import threading
import time
import json
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConnectionMonitor:
    """
    Class for monitoring and managing API connections
    """
    def __init__(self, api=None, notification_manager=None):
        # API client
        self.api = api
        
        # Notification manager
        self.notification_manager = notification_manager
        
        # Monitor settings
        self.health_check_interval = 60  # seconds
        self.max_consecutive_failures = 3
        self.auto_recovery = True
        
        # Monitor state
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Connection state
        self.connection_state = {
            "status": "Not Connected",
            "last_check": None,
            "last_success": None,
            "last_failure": None,
            "consecutive_failures": 0,
            "recovery_attempts": 0,
            "total_checks": 0,
            "total_failures": 0,
            "total_recoveries": 0,
            "uptime_percent": 0,
            "average_response_time": 0
        }
        
        # Connection history
        self.connection_history = []
        self.max_history = 100
        
        # Load saved data
        self._load_data()
        
    def set_api(self, api):
        """
        Set API client
        
        Args:
            api: API client
        """
        self.api = api
        logger.info("Set API client in connection monitor")
        
    def set_notification_manager(self, notification_manager):
        """
        Set notification manager
        
        Args:
            notification_manager: Notification manager
        """
        self.notification_manager = notification_manager
        logger.info("Set notification manager in connection monitor")
        
    def configure(self, health_check_interval=None, max_consecutive_failures=None, auto_recovery=None):
        """
        Configure connection monitor
        
        Args:
            health_check_interval (int, optional): Health check interval in seconds
            max_consecutive_failures (int, optional): Maximum consecutive failures before recovery
            auto_recovery (bool, optional): Whether to attempt automatic recovery
        """
        if health_check_interval is not None:
            self.health_check_interval = max(10, health_check_interval)  # Minimum 10 seconds
            
        if max_consecutive_failures is not None:
            self.max_consecutive_failures = max(1, max_consecutive_failures)
            
        if auto_recovery is not None:
            self.auto_recovery = auto_recovery
            
        logger.info(f"Configured connection monitor: interval={self.health_check_interval}s, max_failures={self.max_consecutive_failures}, auto_recovery={self.auto_recovery}")
        self._save_data()
        
    def start_monitoring(self):
        """
        Start connection monitoring
        
        Returns:
            bool: Success status
        """
        if not self.api:
            logger.error("Cannot start monitoring: API client not set")
            return False
            
        if self.is_monitoring:
            logger.warning("Connection monitoring is already active")
            return False
            
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.is_monitoring = True
        logger.info("Started connection monitoring")
        return True
        
    def stop_monitoring(self):
        """
        Stop connection monitoring
        
        Returns:
            bool: Success status
        """
        if not self.is_monitoring:
            logger.warning("Connection monitoring is not active")
            return False
            
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
            
        self.is_monitoring = False
        logger.info("Stopped connection monitoring")
        return True
        
    def check_connection(self):
        """
        Perform a connection health check
        
        Returns:
            bool: Connection status (True if healthy)
        """
        if not self.api:
            logger.error("Cannot check connection: API client not set")
            return False
            
        # Record check time
        check_time = datetime.now()
        self.connection_state["last_check"] = check_time.strftime("%Y-%m-%d %H:%M:%S")
        self.connection_state["total_checks"] += 1
        
        # Check connection
        try:
            # Time the API call
            start_time = time.time()
            is_authenticated = self.api.is_authenticated()
            response_time = time.time() - start_time
            
            if is_authenticated:
                # Update success state
                self.connection_state["status"] = "Connected"
                self.connection_state["last_success"] = check_time.strftime("%Y-%m-%d %H:%M:%S")
                self.connection_state["consecutive_failures"] = 0
                
                # Update response time
                current_avg = self.connection_state["average_response_time"]
                total_checks = self.connection_state["total_checks"]
                self.connection_state["average_response_time"] = ((current_avg * (total_checks - 1)) + response_time) / total_checks
                
                # Calculate uptime
                successful_checks = total_checks - self.connection_state["total_failures"]
                self.connection_state["uptime_percent"] = (successful_checks / total_checks) * 100 if total_checks > 0 else 0
                
                # Add to history
                self._add_to_history("Connected", response_time)
                
                # Send notification if recovering from failure
                if self.notification_manager and self.connection_state["total_failures"] > 0:
                    self.notification_manager.send_connection_notification(connected=True)
                    
                return True
            else:
                # Update failure state
                self._handle_failure(check_time, "Authentication failed")
                return False
                
        except Exception as e:
            # Update failure state
            self._handle_failure(check_time, str(e))
            return False
            
    def get_connection_state(self):
        """
        Get current connection state
        
        Returns:
            dict: Connection state
        """
        return self.connection_state.copy()
        
    def get_connection_history(self, limit=None):
        """
        Get connection history
        
        Args:
            limit (int, optional): Maximum number of history entries to return
            
        Returns:
            list: Connection history
        """
        history = self.connection_history.copy()
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda h: h.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit:
            history = history[:limit]
            
        return history
        
    def clear_history(self):
        """
        Clear connection history
        """
        self.connection_history = []
        logger.info("Cleared connection history")
        self._save_data()
        
    def _monitor_loop(self):
        """
        Connection monitoring loop
        """
        logger.info("Starting connection monitor loop")
        
        while not self.stop_event.is_set():
            try:
                # Check connection
                connection_ok = self.check_connection()
                
                # Attempt recovery if needed
                if not connection_ok and self.auto_recovery and self.connection_state["consecutive_failures"] >= self.max_consecutive_failures:
                    self._attempt_recovery()
                    
                # Save state
                self._save_data()
                
                # Wait for next check
                for _ in range(int(self.health_check_interval / 2)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error in connection monitor loop: {str(e)}")
                time.sleep(10)  # Sleep on error
                
    def _handle_failure(self, check_time, error_message):
        """
        Handle connection failure
        
        Args:
            check_time (datetime): Time of the check
            error_message (str): Error message
        """
        # Update failure state
        self.connection_state["status"] = "Disconnected"
        self.connection_state["last_failure"] = check_time.strftime("%Y-%m-%d %H:%M:%S")
        self.connection_state["consecutive_failures"] += 1
        self.connection_state["total_failures"] += 1
        
        # Calculate uptime
        total_checks = self.connection_state["total_checks"]
        successful_checks = total_checks - self.connection_state["total_failures"]
        self.connection_state["uptime_percent"] = (successful_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Add to history
        self._add_to_history("Disconnected", 0, error_message)
        
        # Send notification
        if self.notification_manager:
            self.notification_manager.send_connection_notification(connected=False, error_message=error_message)
            
        logger.warning(f"Connection check failed: {error_message}")
        
    def _attempt_recovery(self):
        """
        Attempt to recover connection
        """
        if not self.api:
            logger.error("Cannot attempt recovery: API client not set")
            return False
            
        logger.info("Attempting connection recovery")
        self.connection_state["recovery_attempts"] += 1
        
        try:
            # Attempt to re-login
            recovery_success = self.api.login()
            
            if recovery_success:
                self.connection_state["total_recoveries"] += 1
                logger.info("Connection recovery successful")
                
                # Send notification
                if self.notification_manager:
                    self.notification_manager.send_notification(
                        subject="Connection Recovery Successful",
                        message=f"Successfully recovered connection to Angel One API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        category="connection_issue",
                        importance=4
                    )
                    
                return True
            else:
                logger.error("Connection recovery failed")
                
                # Send notification
                if self.notification_manager:
                    self.notification_manager.send_notification(
                        subject="Connection Recovery Failed",
                        message=f"Failed to recover connection to Angel One API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        category="connection_issue",
                        importance=5
                    )
                    
                return False
                
        except Exception as e:
            logger.error(f"Error during connection recovery: {str(e)}")
            
            # Send notification
            if self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Connection Recovery Error",
                    message=f"Error during connection recovery at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}",
                    category="connection_issue",
                    importance=5
                )
                
            return False
            
    def _add_to_history(self, status, response_time, error_message=None):
        """
        Add entry to connection history
        
        Args:
            status (str): Connection status
            response_time (float): Response time in seconds
            error_message (str, optional): Error message
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "response_time": response_time,
            "error_message": error_message
        }
        
        self.connection_history.append(entry)
        
        # Trim history if needed
        if len(self.connection_history) > self.max_history:
            self.connection_history = self.connection_history[-self.max_history:]
            
    def _save_data(self):
        """
        Save connection monitor data to disk
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            # Prepare data to save
            data = {
                "settings": {
                    "health_check_interval": self.health_check_interval,
                    "max_consecutive_failures": self.max_consecutive_failures,
                    "auto_recovery": self.auto_recovery,
                    "max_history": self.max_history
                },
                "connection_state": self.connection_state,
                "connection_history": self.connection_history,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open("config/connection_monitor.json", "w") as f:
                json.dump(data, f, indent=4)
                
            logger.info("Saved connection monitor data")
        except Exception as e:
            logger.error(f"Error saving connection monitor data: {str(e)}")
            
    def _load_data(self):
        """
        Load connection monitor data from disk
        """
        try:
            # Check if data file exists
            if not os.path.exists("config/connection_monitor.json"):
                logger.info("No connection monitor data file found, using defaults")
                return
                
            # Load data
            with open("config/connection_monitor.json", "r") as f:
                data = json.load(f)
                
            # Apply settings
            if "settings" in data:
                self.health_check_interval = data["settings"].get("health_check_interval", self.health_check_interval)
                self.max_consecutive_failures = data["settings"].get("max_consecutive_failures", self.max_consecutive_failures)
                self.auto_recovery = data["settings"].get("auto_recovery", self.auto_recovery)
                self.max_history = data["settings"].get("max_history", self.max_history)
                
            if "connection_state" in data:
                self.connection_state = data["connection_state"]
                
            if "connection_history" in data:
                self.connection_history = data["connection_history"]
                
            logger.info("Loaded connection monitor data")
        except Exception as e:
            logger.error(f"Error loading connection monitor data: {str(e)}")