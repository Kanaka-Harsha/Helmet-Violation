import cv2
import numpy as np
import mss
import pygetwindow as gw

def capture_window(window_title="Chrome"):
    # Find the window by title (partial match)
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"❌ Window with title '{window_title}' not found.")
        return

    win = windows[0]
    print(f"✅ Found window: {win.title}")

    with mss.mss() as sct:
        cv2.namedWindow("Captured Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Captured Window", 800, 600)
        while True:
            # Update window position in case it moved
            monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
            
            # Ensure width and height are positive (failsafe if minimized)
            if monitor["width"] > 0 and monitor["height"] > 0:
                # Grab the screen region
                sct_img = sct.grab(monitor)

                # Convert to numpy array and BGR for OpenCV
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # Typo check handled below

                cv2.imshow("Captured Window", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_window("Chrome")
