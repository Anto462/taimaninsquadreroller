# Taimanin Squad Auto Reroller

Automated reroller for Taimanin Squad that detects star ratings on character cards and automatically rerolls until you get the desired number of 5-star cards.

## Features

- ✨ Automatically detects 3, 4, and 5-star cards
- 🔄 Continuously rerolls until desired cards are obtained
- 🎯 Multi-template support for different window sizes
- 🐛 Debug mode with screenshot annotations
- ⌨️ Easy exit with ESC or Ctrl+C
- ⚙️ Fully customizable via command-line parameters

## Installation

1. **Install Python 3.7 or higher**
   Download from [python.org](https://www.python.org/downloads/)

2. **Install required libraries**
```bash
   pip install opencv-python numpy pyautogui scikit-learn keyboard pillow
```

3. **Prepare template images**
   - Take a screenshot of a star icon and save as `star_template.png`
   - Take a screenshot of the "Re-recruit" button and save as `recruit_button.png`
   - (Optional) Create additional templates for different window sizes:
     - `star_template2.png`, `star_template3.png`, etc.
     - `recruit_button2.png`, `recruit_button3.png`, etc.

## Usage

### Basic Usage
```bash
python reroller.py
```
This will run with default settings (stop when 3+ five-star cards are found).

### With Parameters
```bash
python reroller.py -max_attempts 100 -min_5_star_cards 2
```

### Common Examples

**Stop after finding 2 five-stars:**
```bash
python reroller.py -min_5_star_cards 2
```

**Enable debug mode to see what's being detected:**
```bash
python reroller.py -debug_mode true
```

**Limit to 50 attempts:**
```bash
python reroller.py -max_attempts 50
```

**Slow game animations (increase delay):**
```bash
python reroller.py -roll_delay 5.0
```

**Combine multiple options:**
```bash
python reroller.py -max_attempts 100 -min_5_star_cards 2 -debug_mode true -roll_delay 4.0
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-max_attempts` | Maximum number of reroll attempts | 9999999 |
| `-min_5_star_cards` | Minimum 5-star cards needed to stop | 3 |
| `-match_threshold` | Template match confidence (0.0-1.0) | 0.80 |
| `-debug_mode` | Save screenshots (`true`/`false`) | false |
| `-roll_delay` | Seconds to wait after clicking reroll | 3.0 |
| `-screenshot_dir` | Where to save debug screenshots | screenshots |
| `-exit_key` | Key to press to exit | esc |

## Keyboard Controls

- **ESC** - Stop the script immediately
- **Ctrl+C** - Also stops the script

## Troubleshooting

### Script not detecting stars/button
1. Enable debug mode: `python reroller.py -debug_mode true`
2. Check `screenshots` folder for detection images
3. Lower the threshold: `python reroller.py -match_threshold 0.70`
4. Recapture your template images at current window size

### False detections
1. Increase the threshold: `python reroller.py -match_threshold 0.85`
2. Ensure templates are clean screenshots without background elements

### Script clicks wrong location
- Recapture the button template image
- Make sure the game window is fully visible and not minimized

## Template Image Tips

- Capture templates with the game at your preferred window size
- Use clean screenshots (no overlapping UI elements)
- Templates should be small (just the star icon or button)
- Create multiple templates if you play at different window sizes

## Output Example
```
[*] Configuration:
    Max Attempts: 9999999
    Min 5-Star Cards: 3
    Match Threshold: 0.8
    Debug Mode: False
    Roll Delay: 3.0s

[*] Loaded 1 star template(s)
[*] Loaded 1 button template(s)

[*] Reroller started! Press 'ESC' or Ctrl+C at any time to stop.

Attempt #1... waiting for button to appear.
Stars per card: [5, 4, 3, 5, 4, 3, 4, 5, 3, 4] | 5-star cards: 3
[SUCCESS] Desired roll achieved! Stopping script.

[DONE] Rerolling complete.
```

## Notes

- Make sure Taimanin Squad is visible on screen (not minimized)
- The script takes control of your mouse - don't move it during operation
- First run with debug mode enabled to verify detection is working
- Run Command Prompt as Administrator if keyboard library has permission issues

## License

Free to use and modify for personal use.
