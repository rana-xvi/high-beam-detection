import cv2
import numpy as np
import pytesseract
import sqlite3
import datetime
import logging
import re
import os
import time
from logging.handlers import TimedRotatingFileHandler
from collections import deque

# --- 1. CONFIGURATION ---

# !! WINDOWS USERS set this to Tesseract-OCR installation path.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Source for the camera. 0 = default webcam.
# Can be a file path ("my_video.mp4") or an RTSP/HTTP stream.
CAMERA_SOURCE = 0

# Time interval (in seconds) between each attempt to run the heavy OCR scan.
OCR_SCAN_INTERVAL_SECONDS = 5.0 
# Time to pause all scanning after a successful log
POST_LOG_COOLDOWN_SECONDS = 5.0 
# Time interval (in seconds) for the live "heartbeat" log
LIVE_LOG_INTERVAL_SECONDS = 5.0

# Brightness threshold (out of 255) to consider a pixel "extremely bright".
MIN_BRIGHTNESS_THRESHOLD = 190

# Minimum number of bright pixels to be considered a "high beam blob".
MIN_AREA_FOR_HIGHBEAM = 40

# A simple regex to validate a potential plate.
# This example is for a generic [ABC-1234] or [ABC1234] format.
# Should change this to match local number plate format.
PLATE_REGEX = r"^[A-Z]{2,4}[ -]?[0-9]{3,4}$"

# --- REAL-TIME LOG BUFFER ---
# Use a deque to store the last N messages for real-time display
MAX_LOG_ENTRIES = 100
LOG_BUFFER = deque(maxlen=MAX_LOG_ENTRIES)
logger = None # Will be initialized in main

# --- 2. LOGGING SETUP ---
def setup_logging():
    """Sets up a logger that rotates its file every 24 hours from launch."""
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Handler automatically rotates the log file.
    # when="S": Rotate based on seconds
    # interval=86400: 86400 seconds = 24 hours
    # Will rotate 24 hours after the script is launched.
    log_handler = TimedRotatingFileHandler(
        "vehicle_logs.txt", 
        when="S", 
        interval=86400, 
        backupCount=30
    )
    
    log_handler.setFormatter(log_formatter)
    
    local_logger = logging.getLogger("HighBeamLogger")
    local_logger.setLevel(logging.INFO)
    local_logger.addHandler(log_handler)
    
    return local_logger

def log_and_display(message, level='INFO'):
    """Logs the message to the file and adds it to the real-time display buffer."""
    
    # 1. Format the message with a simple timestamp for the display buffer
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    display_message = f"[{timestamp}] {message}"
    
    # 2. Add to the global display buffer
    LOG_BUFFER.append(display_message)
    
    # 3. Log to file using Python's logging module
    if logger:
        if level == 'WARNING':
            logger.warning(message)
        elif level == 'ERROR':
            logger.error(message)
        else:
            logger.info(message)

# --- 3. DATABASE SETUP (FOR 24H COOLDOWN) ---
def init_db():
    """Initializes the SQLite database to track logged plates."""
    conn = sqlite3.connect('logged_plates.db')
    cursor = conn.cursor()
    
    # Create a table to store plates and their last-seen timestamp.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plates (
        number_plate TEXT PRIMARY KEY,
        last_seen TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()
    log_and_display("Database initialized.")

def check_and_log_plate(plate_text):
    """
    Checks if a plate was seen in the last 24h.
    If not, logs it to the DB and returns True.
    If it was, returns False.
    """
    conn = sqlite3.connect('logged_plates.db')
    cursor = conn.cursor()
    
    now = datetime.datetime.now()
    twenty_four_hours_ago = now - datetime.timedelta(hours=24)
    
    cursor.execute("SELECT last_seen FROM plates WHERE number_plate = ?", (plate_text,))
    result = cursor.fetchone()
    
    if result:
        last_seen = datetime.datetime.fromisoformat(result[0])
        if last_seen > twenty_four_hours_ago:
            # Already logged in the last 24 hours
            log_and_display(f"Plate {plate_text} already logged in last 24h. Ignoring.", 'WARNING')
            conn.close()
            return False
    
    # New plate, or plate last seen > 24h ago. Log/update it.
    cursor.execute("INSERT OR REPLACE INTO plates (number_plate, last_seen) VALUES (?, ?)", 
                   (plate_text, now.isoformat()))
    conn.commit()
    conn.close()
    
    # This is a new loggable event
    return True

# --- 4. COMPUTER VISION & OCR (IMPROVED OVER 4.0) ---

def scan_for_plate(frame, beam_box):
    """
    Scans a specific Region of Interest (ROI) for a number plate.
    This is much more accurate than scanning the whole frame.
    """
    try:
        frame_h, frame_w = frame.shape[:2]
        bx, by, bw, bh = beam_box

        # Define a Region of Interest (ROI) *below* the high beam.
        center_x = bx + bw // 2
        roi_y1 = by + bh # Start just below the light
        roi_y2 = min(frame_h, by + bh + (bh * 3)) # Look down 4x the light's height
        roi_x1 = max(0, center_x - (bw * 5)) # Look 2x the light's width to the left
        roi_x2 = min(frame_w, center_x + (bw * 5)) # Look 2x the light's width to the right

        # Draw the ROI (blue box) on the main frame for debugging
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        
        # Extract the plate ROI
        plate_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if plate_roi.size == 0:
            return None # ROI is empty

        # --- Pre-processing the ROI for Tesseract ---
        gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize the image (2.5x) to make characters larger and clearer
        scale_factor = 2.5
        resized_roi = cv2.resize(gray_roi, None, fx=scale_factor, fy=scale_factor, 
                                 interpolation=cv2.INTER_CUBIC)

        # Apply a median blur to reduce "salt and pepper" noise
        blurred_roi = cv2.medianBlur(resized_roi, 3)

        # Adaptive thresholding to get a clean binary image
        thresh_roi = cv2.adaptiveThreshold(blurred_roi, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)

        # Show the pre-processed image in a new window for debugging
        cv2.imshow("OCR Pre-processing", thresh_roi)

        # --- Tesseract Configuration ---
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        text = pytesseract.image_to_string(thresh_roi, config=custom_config)
        
        # --- Real-Time Log: Raw OCR Result ---
        raw_text = text.strip()
        if raw_text:
            log_and_display(f"OCR raw text detected: {raw_text}")
        # --- End of real-time log step ---
        
        # Clean and validate the text
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if re.match(PLATE_REGEX, cleaned_text):
            log_and_display(f"Potential plate found: {cleaned_text}")
            return cleaned_text
                
    except Exception as e:
        log_and_display(f"Error during OCR: {e}", 'ERROR')
    
    return None

def update_log_window(buffer):
    """Draws the log buffer onto a dedicated OpenCV window."""
    # Create a black image (600 width, 400 height)
    log_image = np.zeros((400, 600, 3), dtype="uint8")
    
    y_start = 30
    for i, message in enumerate(buffer):
        # Draw the message in white
        cv2.putText(log_image, message, (10, y_start + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
    cv2.imshow("Real-time Log", log_image)


def main():
    """Main application loop."""
    global logger
    logger = setup_logging()
    init_db()
    
    print("Starting high beam detector... (Press 'q' to quit)")
    
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera source {CAMERA_SOURCE}. Exiting.")
        return

    # Non-blocking cooldown timers
    last_ocr_scan_time = 1.0      # Time when the last OCR scan was *started*
    post_log_cooldown_until = 2.0 # Time until the system can log a new plate
    last_live_log_time = time.time() # For the 5-sec heartbeat log

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or camera disconnected. Exiting.")
            break
            
        current_time = time.time()
        high_beam_detected = False
        beam_bounding_box = None
        status_message = "ACTIVE"
        status_color = (0, 255, 0) # Green for ACTIVE
        
        try:
            # 1. High Beam Detection (No AI)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply a binary threshold for *very bright* pixels
            _, bright_mask = cv2.threshold(gray, MIN_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # Find contours (blobs) in the bright mask
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA_FOR_HIGHBEAM:
                    # Found a blob large and bright enough
                    high_beam_detected = True
                    beam_bounding_box = cv2.boundingRect(cnt) # Store the box
                    
                    # Draw a red rectangle for visualization
                    x, y, w, h = beam_bounding_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    status_message = "HIGH BEAM DETECTED"
                    status_color = (0, 165, 255) # Orange
                    break # One blob is enough to trigger
            
            # 2. If high beam detected, try to scan for a plate (Throttled)
            if high_beam_detected and beam_bounding_box:
                
                # Check if we are in the post-log cooldown
                if current_time > post_log_cooldown_until:
                    
                    # Check if we are in the 5-second scan interval
                    if (current_time - last_ocr_scan_time) > OCR_SCAN_INTERVAL_SECONDS:
                        last_ocr_scan_time = current_time # Start scan, update timer
                        
                        status_message = "SCANNING..."
                        status_color = (0, 255, 255) # Yellow
                        
                        # Pass the frame AND the beam's box to the new OCR func
                        plate_text = scan_for_plate(frame, beam_bounding_box)
                        
                        if plate_text:
                            # 3. If plate found, check 24h rule and log
                            if check_and_log_plate(plate_text):
                                # Log it to the file and display buffer
                                log_and_display(f"VIOLATION LOGGED: {plate_text} (New Record)")
                                print(f"** VIOLATION LOGGED: {plate_text} **")
                                
                                # Apply 5-second buffer to prevent re-logging same car
                                post_log_cooldown_until = current_time + POST_LOG_COOLDOWN_SECONDS
                                status_message = f"LOGGED: {plate_text}"
                                status_color = (0, 0, 255) # Red
                            else:
                                status_message = f"SEEN: {plate_text} (24h Rule)"
                        else:
                            # Scan ran, no plate found
                            status_message = "HIGH BEAM (Plate not read)"
                    else:
                        # High beam seen, but waiting for scan interval
                        remaining = OCR_SCAN_INTERVAL_SECONDS - (current_time - last_ocr_scan_time)
                        status_message = f"HIGH BEAM (Scan in {remaining:.1f}s)"
                else:
                    # In post-log cooldown
                    remaining = post_log_cooldown_until - current_time
                    status_message = f"LOG COOLDOWN ({remaining:.1f}s)"
                    status_color = (0, 0, 255) # Red
            
            # 3. Live "Heartbeat" Logging
            if (current_time - last_live_log_time) > LIVE_LOG_INTERVAL_SECONDS:
                log_and_display(f"Status: {status_message}")
                last_live_log_time = current_time

            # 4. Show the output and update log window
            # Draw the status message on the frame
            cv2.putText(frame, status_message, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA) # Black outline
            cv2.putText(frame, status_message, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA) # Colored text
            
            cv2.imshow("High Beam Detector", frame)
            update_log_window(LOG_BUFFER) # Update the new log window

        except Exception as e:
            # Use print for console debugging, and log_and_display for file/window
            print(f"Error in main loop: {e}") 
            log_and_display(f"CRITICAL ERROR in main loop: {e}", 'ERROR')
            
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q'. Exiting.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    try:
        cv2.destroyWindow("OCR Pre-processing") # Clean up the debug window
        cv2.destroyWindow("Real-time Log") # Clean up the new log window
    except:
        pass # Ignore error if windows were never opened
    print("Application shut down.")

if __name__ == "__main__":
    main()
