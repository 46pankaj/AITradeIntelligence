import logging
import threading
import time
import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from queue import Queue
from twilio.rest import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Class for managing and sending notifications through various channels
    """
    def __init__(self):
        # Email settings
        self.email_enabled = False
        self.email_from = ""
        self.email_password = ""
        self.email_smtp_server = "smtp.gmail.com"
        self.email_smtp_port = 587
        self.email_recipients = []
        
        # SMS settings (Twilio)
        self.sms_enabled = False
        self.twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone_number = os.environ.get("TWILIO_PHONE_NUMBER", "")
        self.sms_recipients = []
        
        # Notification queue and worker thread
        self.notification_queue = Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Notification log
        self.notification_log = []
        self.max_log_size = 100
        
        # Notification categories and their importance level (1-5)
        self.categories = {
            "trade_execution": 5,
            "profit_taking": 5,
            "stop_loss": 5,
            "strategy_change": 4,
            "market_alert": 4,
            "authentication": 5,
            "connection_issue": 5,
            "system_status": 3,
            "general": 2
        }
        
        # Notification importance thresholds
        self.email_importance_threshold = 3
        self.sms_importance_threshold = 4
        
        # Load saved settings
        self._load_settings()
        
    def configure_email(self, enabled, from_email, password, recipients, smtp_server=None, smtp_port=None):
        """
        Configure email notifications
        
        Args:
            enabled (bool): Enable or disable email notifications
            from_email (str): Email address to send from
            password (str): Email password or app password
            recipients (list): List of email recipients
            smtp_server (str, optional): SMTP server address
            smtp_port (int, optional): SMTP server port
        """
        self.email_enabled = enabled
        self.email_from = from_email
        self.email_password = password
        self.email_recipients = recipients
        
        if smtp_server:
            self.email_smtp_server = smtp_server
            
        if smtp_port:
            self.email_smtp_port = smtp_port
            
        logger.info(f"Email notifications {'enabled' if enabled else 'disabled'}")
        self._save_settings()
        
    def configure_sms(self, enabled, recipients):
        """
        Configure SMS notifications
        
        Args:
            enabled (bool): Enable or disable SMS notifications
            recipients (list): List of phone numbers to send SMS to
        """
        if enabled and (not self.twilio_account_sid or not self.twilio_auth_token or not self.twilio_phone_number):
            logger.error("Cannot enable SMS: Twilio credentials not set")
            return False
            
        self.sms_enabled = enabled
        self.sms_recipients = recipients
        
        logger.info(f"SMS notifications {'enabled' if enabled else 'disabled'}")
        self._save_settings()
        return True
        
    def set_importance_thresholds(self, email_threshold, sms_threshold):
        """
        Set importance thresholds for notifications
        
        Args:
            email_threshold (int): Minimum importance level for email notifications (1-5)
            sms_threshold (int): Minimum importance level for SMS notifications (1-5)
        """
        self.email_importance_threshold = max(1, min(5, email_threshold))
        self.sms_importance_threshold = max(1, min(5, sms_threshold))
        
        logger.info(f"Set notification thresholds: Email={self.email_importance_threshold}, SMS={self.sms_importance_threshold}")
        self._save_settings()
        
    def start_notification_worker(self):
        """
        Start the notification worker thread
        
        Returns:
            bool: Success status
        """
        if self.worker_thread and self.worker_thread.is_alive():
            logger.warning("Notification worker is already running")
            return False
            
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("Started notification worker")
        return True
        
    def stop_notification_worker(self):
        """
        Stop the notification worker thread
        
        Returns:
            bool: Success status
        """
        if not self.worker_thread or not self.worker_thread.is_alive():
            logger.warning("Notification worker is not running")
            return False
            
        self.stop_event.set()
        self.worker_thread.join(timeout=10)
        
        logger.info("Stopped notification worker")
        return True
        
    def send_notification(self, subject, message, category="general", importance=None, data=None):
        """
        Send a notification
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            category (str, optional): Notification category
            importance (int, optional): Importance level (1-5). If None, use category default.
            data (dict, optional): Additional data for the notification
            
        Returns:
            str: Notification ID
        """
        # Generate a unique notification ID
        notification_id = f"NOTIF-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.notification_log)}"
        
        # Get importance level from category if not specified
        if importance is None:
            importance = self.categories.get(category, 2)
        else:
            importance = max(1, min(5, importance))
            
        # Create notification object
        notification = {
            "id": notification_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject": subject,
            "message": message,
            "category": category,
            "importance": importance,
            "data": data or {}
        }
        
        # Add to notification log
        self._add_to_log(notification)
        
        # Add to queue for sending
        self.notification_queue.put(notification)
        
        logger.info(f"Queued notification: {subject} (ID: {notification_id})")
        return notification_id
        
    def send_trade_notification(self, trade_data):
        """
        Send a notification about a trade
        
        Args:
            trade_data (dict): Trade data
            
        Returns:
            str: Notification ID
        """
        trade_type = trade_data.get("type", "UNKNOWN")
        symbol = trade_data.get("symbol", "UNKNOWN")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        
        subject = f"Trade Executed: {trade_type} {symbol}"
        message = f"""Trade Details:
Symbol: {symbol}
Type: {trade_type}
Quantity: {quantity}
Price: ₹{price:,.2f}
Value: ₹{price * quantity:,.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_notification(
            subject=subject,
            message=message,
            category="trade_execution",
            data=trade_data
        )
        
    def send_profit_notification(self, trade_data, profit_amount, profit_percent):
        """
        Send a notification about profit taking
        
        Args:
            trade_data (dict): Trade data
            profit_amount (float): Profit amount
            profit_percent (float): Profit percentage
            
        Returns:
            str: Notification ID
        """
        symbol = trade_data.get("symbol", "UNKNOWN")
        
        subject = f"Profit Booked: {symbol} (+{profit_percent:.2f}%)"
        message = f"""Profit Details:
Symbol: {symbol}
Profit Amount: ₹{profit_amount:,.2f}
Profit Percentage: {profit_percent:.2f}%
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_notification(
            subject=subject,
            message=message,
            category="profit_taking",
            data={
                "trade": trade_data,
                "profit_amount": profit_amount,
                "profit_percent": profit_percent
            }
        )
        
    def send_stop_loss_notification(self, trade_data, loss_amount, loss_percent):
        """
        Send a notification about stop loss
        
        Args:
            trade_data (dict): Trade data
            loss_amount (float): Loss amount
            loss_percent (float): Loss percentage
            
        Returns:
            str: Notification ID
        """
        symbol = trade_data.get("symbol", "UNKNOWN")
        
        subject = f"Stop Loss Triggered: {symbol} ({loss_percent:.2f}%)"
        message = f"""Stop Loss Details:
Symbol: {symbol}
Loss Amount: ₹{loss_amount:,.2f}
Loss Percentage: {loss_percent:.2f}%
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_notification(
            subject=subject,
            message=message,
            category="stop_loss",
            data={
                "trade": trade_data,
                "loss_amount": loss_amount,
                "loss_percent": loss_percent
            }
        )
        
    def send_strategy_notification(self, strategy_data):
        """
        Send a notification about a strategy change
        
        Args:
            strategy_data (dict): Strategy data
            
        Returns:
            str: Notification ID
        """
        strategy_name = strategy_data.get("name", "UNKNOWN")
        symbol = strategy_data.get("symbol", "UNKNOWN")
        
        subject = f"Strategy Generated: {strategy_name} for {symbol}"
        message = f"""Strategy Details:
Name: {strategy_name}
Symbol: {symbol}
Type: {strategy_data.get('type', 'UNKNOWN')}
Timeframe: {strategy_data.get('timeframe', 'UNKNOWN')}
Status: {strategy_data.get('status', 'UNKNOWN')}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_notification(
            subject=subject,
            message=message,
            category="strategy_change",
            data=strategy_data
        )
        
    def send_authentication_notification(self, success, error_message=None):
        """
        Send a notification about authentication
        
        Args:
            success (bool): Whether authentication was successful
            error_message (str, optional): Error message if authentication failed
            
        Returns:
            str: Notification ID
        """
        if success:
            subject = "Authentication Successful"
            message = f"Successfully authenticated with Angel One API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            category = "authentication"
            importance = 3
        else:
            subject = "Authentication Failed"
            message = f"""Failed to authenticate with Angel One API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {error_message or 'Unknown error'}
"""
            category = "authentication"
            importance = 5
            
        return self.send_notification(
            subject=subject,
            message=message,
            category=category,
            importance=importance,
            data={"success": success, "error": error_message}
        )
        
    def send_connection_notification(self, connected, error_message=None):
        """
        Send a notification about connection status
        
        Args:
            connected (bool): Whether connection is established
            error_message (str, optional): Error message if connection failed
            
        Returns:
            str: Notification ID
        """
        if connected:
            subject = "Connection Established"
            message = f"Successfully connected to Angel One API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            category = "connection_issue"
            importance = 3
        else:
            subject = "Connection Lost"
            message = f"""Lost connection to Angel One API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {error_message or 'Unknown error'}
"""
            category = "connection_issue"
            importance = 5
            
        return self.send_notification(
            subject=subject,
            message=message,
            category=category,
            importance=importance,
            data={"connected": connected, "error": error_message}
        )
        
    def get_notification_log(self, limit=None, category=None, min_importance=None):
        """
        Get notification log
        
        Args:
            limit (int, optional): Maximum number of notifications to return
            category (str, optional): Filter by category
            min_importance (int, optional): Minimum importance level
            
        Returns:
            list: List of notifications
        """
        # Make a copy of the log
        log = self.notification_log.copy()
        
        # Apply filters
        if category:
            log = [n for n in log if n["category"] == category]
            
        if min_importance:
            log = [n for n in log if n["importance"] >= min_importance]
            
        # Sort by timestamp (newest first)
        log.sort(key=lambda n: n["timestamp"], reverse=True)
        
        # Apply limit
        if limit:
            log = log[:limit]
            
        return log
        
    def clear_notification_log(self):
        """
        Clear notification log
        """
        self.notification_log = []
        logger.info("Cleared notification log")
        
    def _notification_worker(self):
        """
        Worker thread for sending notifications
        """
        logger.info("Starting notification worker")
        
        while not self.stop_event.is_set():
            try:
                # Try to get a notification from the queue with a 1-second timeout
                try:
                    notification = self.notification_queue.get(timeout=1)
                except Exception:
                    # No notification in queue, continue
                    continue
                    
                # Extract notification details
                subject = notification["subject"]
                message = notification["message"]
                importance = notification["importance"]
                
                # Send via appropriate channels based on importance
                sent_email = False
                sent_sms = False
                
                # Send email if enabled and importance meets threshold
                if self.email_enabled and importance >= self.email_importance_threshold:
                    try:
                        self._send_email(subject, message)
                        sent_email = True
                    except Exception as e:
                        logger.error(f"Error sending email notification: {str(e)}")
                        
                # Send SMS if enabled and importance meets threshold
                if self.sms_enabled and importance >= self.sms_importance_threshold:
                    try:
                        self._send_sms(subject, message)
                        sent_sms = True
                    except Exception as e:
                        logger.error(f"Error sending SMS notification: {str(e)}")
                        
                logger.info(f"Processed notification: {subject} (Email: {sent_email}, SMS: {sent_sms})")
                
                # Mark as done
                self.notification_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in notification worker: {str(e)}")
                time.sleep(1)  # Sleep on error
                
    def _send_email(self, subject, message):
        """
        Send an email notification
        
        Args:
            subject (str): Email subject
            message (str): Email message
        """
        if not self.email_enabled or not self.email_recipients:
            logger.warning("Email notifications are disabled or no recipients configured")
            return
            
        # Create MIMEText object
        msg = MIMEMultipart()
        msg["From"] = self.email_from
        msg["To"] = ", ".join(self.email_recipients)
        msg["Subject"] = f"TradeAI: {subject}"
        
        # Add message body
        msg.attach(MIMEText(message, "plain"))
        
        # Connect to SMTP server and send email
        try:
            server = smtplib.SMTP(self.email_smtp_server, self.email_smtp_port)
            server.starttls()
            server.login(self.email_from, self.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent email notification: {subject}")
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            raise
            
    def _send_sms(self, subject, message):
        """
        Send an SMS notification
        
        Args:
            subject (str): SMS subject
            message (str): SMS message
        """
        if not self.sms_enabled or not self.sms_recipients:
            logger.warning("SMS notifications are disabled or no recipients configured")
            return
            
        # Check if Twilio credentials are available
        if not self.twilio_account_sid or not self.twilio_auth_token or not self.twilio_phone_number:
            logger.error("Twilio credentials not configured")
            return
            
        # Create Twilio client
        try:
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            # Combine subject and message
            sms_text = f"{subject}\n\n{message}"
            
            # Truncate if too long (Twilio has a limit)
            if len(sms_text) > 1500:
                sms_text = sms_text[:1497] + "..."
                
            # Send to all recipients
            for recipient in self.sms_recipients:
                client.messages.create(
                    body=sms_text,
                    from_=self.twilio_phone_number,
                    to=recipient
                )
                
            logger.info(f"Sent SMS notification: {subject}")
        except Exception as e:
            logger.error(f"Error sending SMS: {str(e)}")
            raise
            
    def _add_to_log(self, notification):
        """
        Add a notification to the log
        
        Args:
            notification (dict): Notification object
        """
        self.notification_log.append(notification)
        
        # Trim log if it exceeds max size
        if len(self.notification_log) > self.max_log_size:
            # Sort by timestamp (oldest first)
            self.notification_log.sort(key=lambda n: n["timestamp"])
            
            # Remove oldest notifications
            self.notification_log = self.notification_log[-self.max_log_size:]
            
    def _save_settings(self):
        """
        Save notification settings to disk
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            # Prepare settings to save
            settings = {
                "email": {
                    "enabled": self.email_enabled,
                    "from": self.email_from,
                    "smtp_server": self.email_smtp_server,
                    "smtp_port": self.email_smtp_port,
                    "recipients": self.email_recipients
                },
                "sms": {
                    "enabled": self.sms_enabled,
                    "recipients": self.sms_recipients
                },
                "thresholds": {
                    "email": self.email_importance_threshold,
                    "sms": self.sms_importance_threshold
                },
                "categories": self.categories,
                "max_log_size": self.max_log_size
            }
            
            # Save to file
            with open("config/notification_settings.json", "w") as f:
                json.dump(settings, f, indent=4)
                
            logger.info("Saved notification settings")
        except Exception as e:
            logger.error(f"Error saving notification settings: {str(e)}")
            
    def _load_settings(self):
        """
        Load notification settings from disk
        """
        try:
            # Check if settings file exists
            if not os.path.exists("config/notification_settings.json"):
                logger.info("No notification settings file found")
                return
                
            # Load settings
            with open("config/notification_settings.json", "r") as f:
                settings = json.load(f)
                
            # Apply settings
            if "email" in settings:
                self.email_enabled = settings["email"].get("enabled", False)
                self.email_from = settings["email"].get("from", "")
                self.email_smtp_server = settings["email"].get("smtp_server", "smtp.gmail.com")
                self.email_smtp_port = settings["email"].get("smtp_port", 587)
                self.email_recipients = settings["email"].get("recipients", [])
                
            if "sms" in settings:
                self.sms_enabled = settings["sms"].get("enabled", False)
                self.sms_recipients = settings["sms"].get("recipients", [])
                
            if "thresholds" in settings:
                self.email_importance_threshold = settings["thresholds"].get("email", 3)
                self.sms_importance_threshold = settings["thresholds"].get("sms", 4)
                
            if "categories" in settings:
                self.categories.update(settings["categories"])
                
            if "max_log_size" in settings:
                self.max_log_size = settings["max_log_size"]
                
            logger.info("Loaded notification settings")
        except Exception as e:
            logger.error(f"Error loading notification settings: {str(e)}")