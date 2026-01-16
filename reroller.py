import cv2
import numpy as np
import pyautogui
import time
import os
import sys
import signal
import glob
import argparse
from sklearn.cluster import DBSCAN
import keyboard  # pip install keyboard

# -----------------------------
# COMMAND LINE ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(
    description='Automated reroller for gacha game recruitment',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python reroller.py
  python reroller.py -max_attempts 50 -min_5_star_cards 2
  python reroller.py -debug_mode true -match_threshold 0.75
  python reroller.py -roll_delay 5 -max_attempts 100
    """
)

parser.add_argument('-max_attempts', type=int, default=9999999,
                    help='Maximum number of reroll attempts (default: 9999999)')
parser.add_argument('-min_5_star_cards', type=int, default=3,
                    help='Minimum number of 5-star cards required to stop (default: 3)')
parser.add_argument('-match_threshold', type=float, default=0.85,
                    help='Template matching confidence threshold 0.0-1.0 (default: 0.85)')
parser.add_argument('-debug_mode', type=str, default='false',
                    choices=['true', 'false'],
                    help='Enable debug mode to save screenshots (default: false)')
parser.add_argument('-roll_delay', type=float, default=3.0,
                    help='Seconds to wait after clicking reroll button (default: 3.0)')
parser.add_argument('-screenshot_dir', type=str, default='screenshots',
                    help='Directory to save debug screenshots (default: screenshots)')
parser.add_argument('-exit_key', type=str, default='esc',
                    help='Key to press to exit the script (default: esc)')

args = parser.parse_args()

# -----------------------------
# PARAMETERS
# -----------------------------
STAR_TEMPLATE_PATTERN = "star_template*.png"  # Matches star_template.png, star_template2.png, etc.
BUTTON_TEMPLATE_PATTERN = "recruit_button*.png"  # Matches recruit_button.png, recruit_button2.png, etc.

ROLL_DELAY = args.roll_delay
MAX_ATTEMPTS = args.max_attempts
MIN_5_STAR_CARDS = args.min_5_star_cards
MATCH_THRESHOLD = args.match_threshold
DEBUG_MODE = args.debug_mode.lower() == 'true'
SCREENSHOT_DIR = args.screenshot_dir
EXIT_KEY = args.exit_key

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Print current configuration
print("[*] Configuration:")
print(f"    Max Attempts: {MAX_ATTEMPTS}")
print(f"    Min 5-Star Cards: {MIN_5_STAR_CARDS}")
print(f"    Match Threshold: {MATCH_THRESHOLD}")
print(f"    Debug Mode: {DEBUG_MODE}")
print(f"    Roll Delay: {ROLL_DELAY}s")
print(f"    Screenshot Dir: {SCREENSHOT_DIR}")
print(f"    Exit Key: {EXIT_KEY.upper()}")
print()

# -----------------------------
# LOAD ALL TEMPLATES
# -----------------------------
def load_templates(pattern):
    """Load all template images matching the pattern."""
    template_files = glob.glob(pattern)
    templates = []
    for file in template_files:
        template = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates.append((file, template))
            print(f"[*] Loaded template: {file} (size: {template.shape})")
        else:
            print(f"[WARNING] Could not load template: {file}")
    return templates

# Load all star and button templates at startup
STAR_TEMPLATES = load_templates(STAR_TEMPLATE_PATTERN)
BUTTON_TEMPLATES = load_templates(BUTTON_TEMPLATE_PATTERN)

if not STAR_TEMPLATES:
    print(f"[ERROR] No star templates found matching pattern: {STAR_TEMPLATE_PATTERN}")
    sys.exit(1)

if not BUTTON_TEMPLATES:
    print(f"[ERROR] No button templates found matching pattern: {BUTTON_TEMPLATE_PATTERN}")
    sys.exit(1)

print(f"[*] Loaded {len(STAR_TEMPLATES)} star template(s)")
print(f"[*] Loaded {len(BUTTON_TEMPLATES)} button template(s)\n")

# -----------------------------
# EXIT FLAG & SIGNAL HANDLING
# -----------------------------
should_exit = False

def on_exit_key():
    global should_exit
    should_exit = True
    print("\n[!] ESC pressed! Stopping script...")

def signal_handler(sig, frame):
    global should_exit
    should_exit = True
    print("\n[!] Ctrl+C pressed! Stopping script...")
    keyboard.unhook_all()
    sys.exit(0)

# Set up the exit key listener and Ctrl+C handler
keyboard.on_press_key(EXIT_KEY, lambda _: on_exit_key())
signal.signal(signal.SIGINT, signal_handler)

# -----------------------------
# NMS FUNCTION
# -----------------------------
def nms(points, scores, template_size, overlap_thresh=0.3):
    if len(points) == 0:
        return []

    boxes = []
    for (x, y) in points:
        boxes.append([x, y, x + template_size[0], y + template_size[1]])
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]
    return [points[i] for i in keep]

# -----------------------------
# MULTI-TEMPLATE MATCHING
# -----------------------------
def find_best_template_match(screen_gray, templates, threshold=MATCH_THRESHOLD):
    """
    Try all templates and return the best match that exceeds threshold.
    Returns: (best_points, best_template_shape, best_template_name, best_max_confidence)
    """
    best_result = None
    best_confidence = 0
    best_points = []
    best_template_shape = None
    best_template_name = None
    
    for template_name, template_gray in templates:
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        points = list(zip(*locations[::-1]))
        
        if points:
            scores = [result[y, x] for x, y in points]
            max_score = max(scores)
            
            if max_score > best_confidence:
                best_confidence = max_score
                filtered = nms(points, scores, template_gray.shape, overlap_thresh=0.3)
                best_points = filtered
                best_template_shape = template_gray.shape
                best_template_name = template_name
                best_result = result
    
    return best_points, best_template_shape, best_template_name, best_confidence

def find_stars_on_screen():
    screenshot = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    points, template_shape, template_name, confidence = find_best_template_match(
        screen_gray, STAR_TEMPLATES, threshold=MATCH_THRESHOLD
    )
    
    if DEBUG_MODE and template_name:
        print(f"  [DEBUG] Using star template: {template_name} (confidence: {confidence:.3f})")
    
    # DEBUG: draw green rectangles for stars
    if DEBUG_MODE and template_shape:
        for x, y in points:
            cv2.rectangle(screen, (x, y), 
                         (x + template_shape[1], y + template_shape[0]), 
                         (0, 255, 0), 2)
    
    return points, template_shape, screen

def find_button_on_screen(screen=None, save_debug=False):
    screenshot = pyautogui.screenshot()
    if screen is None:
        screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    # Try all button templates
    best_match = None
    best_confidence = 0
    best_template_name = None
    best_location = None
    best_size = None
    
    for template_name, template_gray in BUTTON_TEMPLATES:
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_template_name = template_name
            best_location = max_loc
            best_size = template_gray.shape[:2]
            best_match = result
    
    # DEBUG: Save comparison images
    if save_debug:
        cv2.imwrite(f"{SCREENSHOT_DIR}/debug_screen_gray.png", screen_gray)
        for template_name, template_gray in BUTTON_TEMPLATES:
            safe_name = os.path.basename(template_name).replace('.png', '')
            cv2.imwrite(f"{SCREENSHOT_DIR}/debug_template_{safe_name}.png", template_gray)
        
        if best_match is not None:
            # Save heatmap of best match
            heatmap = cv2.normalize(best_match, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(f"{SCREENSHOT_DIR}/debug_heatmap.png", heatmap_color)
        
        print(f"[DEBUG] Saved debug images to {SCREENSHOT_DIR}/")
        print(f"[DEBUG] Best template: {best_template_name}")
        print(f"[DEBUG] Best confidence: {best_confidence:.3f}")
        print(f"[DEBUG] Screen size: {screen_gray.shape}")
    
    # DEBUG: show max confidence value
    if DEBUG_MODE and best_confidence < MATCH_THRESHOLD:
        print(f"  [DEBUG] Button match confidence: {best_confidence:.3f} (threshold: {MATCH_THRESHOLD})")
        if best_template_name:
            print(f"  [DEBUG] Best matching template: {best_template_name}")
    
    if best_confidence >= MATCH_THRESHOLD:
        if DEBUG_MODE:
            print(f"  [DEBUG] Button found using template: {best_template_name}")
        
        h, w = best_size
        center_x = best_location[0] + w // 2
        center_y = best_location[1] + h // 2
        
        if DEBUG_MODE:
            cv2.rectangle(screen, 
                         (best_location[0], best_location[1]), 
                         (best_location[0]+w, best_location[1]+h), 
                         (0, 0, 255), 3)
        
        return (center_x, center_y), screen
    
    return None, screen

# -----------------------------
# IMPROVED CLUSTERING & COUNTING
# -----------------------------
def count_stars_per_card(star_points, template_shape):
    if not star_points or not template_shape:
        return []
    
    # Sort all points by X coordinate (left to right)
    sorted_points = sorted(star_points, key=lambda p: p[0])
    
    # Group stars into cards based on horizontal gaps
    cards = []
    current_card = [sorted_points[0]]
    
    # Calculate expected spacing between stars within a card
    # Stars within a card should be close together
    star_width = template_shape[1]
    card_gap_threshold = star_width * 3  # Gap between cards is much larger
    
    for i in range(1, len(sorted_points)):
        prev_x = sorted_points[i-1][0]
        curr_x = sorted_points[i][0]
        
        # If gap is large, it's a new card
        if curr_x - prev_x > card_gap_threshold:
            cards.append(current_card)
            current_card = [sorted_points[i]]
        else:
            current_card.append(sorted_points[i])
    
    # Don't forget the last card
    if current_card:
        cards.append(current_card)
    
    # Count stars in each card by looking at unique horizontal positions
    stars_per_card = []
    for card_stars in cards:
        # Sort by X within this card
        card_stars = sorted(card_stars, key=lambda p: p[0])
        
        # Group stars that are at the same horizontal position (same column)
        unique_x_positions = []
        for x, y in card_stars:
            # Check if this X position is already recorded (within tolerance)
            is_new = True
            for existing_x in unique_x_positions:
                if abs(x - existing_x) < star_width * 0.5:
                    is_new = False
                    break
            if is_new:
                unique_x_positions.append(x)
        
        stars_per_card.append(len(unique_x_positions))
    
    return stars_per_card

# -----------------------------
# MAIN LOOP
# -----------------------------
print(f"[*] Reroller started! Press '{EXIT_KEY.upper()}' or Ctrl+C at any time to stop.\n")

attempt = 0
try:
    while attempt < MAX_ATTEMPTS:
        # Check if user wants to exit
        if should_exit:
            print("[X] Script stopped by user.")
            break
            
        attempt += 1
        print(f"\nAttempt #{attempt}... waiting for button to appear.")

        # Wait for button with timeout
        button_pos = None
        screen = None
        wait_iterations = 0
        max_wait_iterations = 60  # 30 seconds (60 * 0.5s)
        
        while button_pos is None:
            if should_exit:
                print("[X] Script stopped by user.")
                break
            
            # Save debug images on first check
            save_debug = (wait_iterations == 0 and attempt == 1)
            button_pos, screen = find_button_on_screen(screen, save_debug=save_debug)
            
            if button_pos is None:
                wait_iterations += 1
                if wait_iterations >= max_wait_iterations:
                    print("[!] Timeout: Button not found after 30 seconds.")
                    print("[!] Tips:")
                    print("    - Make sure the game window is visible")
                    print("    - Add more button templates (recruit_button2.png, etc.)")
                    print("    - Look at the debug images in the screenshots folder")
                    print("    - Try recapturing templates at the current window size")
                    should_exit = True
                    break
                time.sleep(0.5)
        
        if should_exit:
            break
        
        # Scan stars
        star_points, template_shape, annotated_screen = find_stars_on_screen()
        stars_per_card = count_stars_per_card(star_points, template_shape)
        total_5_star_cards = sum(1 for s in stars_per_card if s >= 5)
        
        print(f"Stars per card: {stars_per_card} | 5-star cards: {total_5_star_cards}")
        
        # Save debug screenshot with card labels
        if DEBUG_MODE:
            # Add text labels for each detected card count
            for idx, count in enumerate(stars_per_card):
                cv2.putText(annotated_screen, f"Card {idx+1}: {count} stars", 
                           (10, 30 + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 2)
            
            filename = f"{SCREENSHOT_DIR}/attempt_{attempt:03d}.png"
            cv2.imwrite(filename, annotated_screen)
            print(f"Debug screenshot saved: {filename}")

        if total_5_star_cards >= MIN_5_STAR_CARDS:
            print("[SUCCESS] Desired roll achieved! Stopping script.")
            break

        if should_exit:
            break
            
        print(f"Clicking re-recruit button at {button_pos}...")
        pyautogui.click(*button_pos)
        
        # Wait with periodic exit checks
        for _ in range(int(ROLL_DELAY * 10)):
            if should_exit:
                break
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[!] Ctrl+C detected! Exiting...")
finally:
    print("\n[DONE] Rerolling complete.")
    keyboard.unhook_all()  # Clean up keyboard listener