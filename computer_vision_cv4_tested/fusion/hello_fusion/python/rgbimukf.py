import cv2
import numpy as np

def make_kalman(dt: float):
    kf = cv2.KalmanFilter(4, 2, 2)  # state=4, measurement=2, control=2

    # State: [x, y, vx, vy]
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0 ],
        [0, 1, 0,  dt],
        [0, 0, 1,  0 ],
        [0, 0, 0,  1 ],
    ], dtype=np.float32)

    # Control model for acceleration: x += 0.5*a*dt^2, vx += a*dt
    kf.controlMatrix = np.array([
        [0.5*dt*dt, 0],
        [0, 0.5*dt*dt],
        [dt, 0],
        [0, dt],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # Noise tuning (adjust to see behavior)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2   # model uncertainty
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0  # vision measurement noise
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    return kf

def detect_colored_object(frame_bgr):
    """
    Example 'vision sensor': detect a bright green object.
    Returns (x, y) center in pixels or None if not found.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Green range (tweak as needed)
    lower = np.array([40, 80, 80], dtype=np.uint8)
    upper = np.array([80, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 7)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 150:
        return None, mask

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try changing index or use a video file.")

    dt = 1.0 / 30.0
    kf = make_kalman(dt)

    # Initialize with center-ish
    initialized = False

    # Simulated IMU parameters (for demo): assume small random acceleration bias + noise
    imu_bias = np.array([0.0, 0.0], dtype=np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- "IMU sensor": acceleration estimate (simulated) ---
        # In real life, you'd read ax, ay from an IMU. Here we just make something plausible.
        imu_bias += np.random.randn(2).astype(np.float32) * 0.01  # slowly drifting bias
        a = imu_bias + np.random.randn(2).astype(np.float32) * 0.2  # noisy accel

        # Predict with control input (acceleration)
        pred = kf.predict(control=a.reshape(2,1))
        px, py = float(pred[0]), float(pred[1])

        # --- "Vision sensor": measurement from RGB ---
        meas, mask = detect_colored_object(frame)

        if meas is not None:
            mx, my = meas
            z = np.array([[np.float32(mx)], [np.float32(my)]])
            if not initialized:
                # Set initial state from first measurement
                kf.statePost = np.array([[mx], [my], [0], [0]], dtype=np.float32)
                initialized = True
            else:
                kf.correct(z)

            cv2.circle(frame, (mx, my), 6, (255, 255, 255), -1)  # measured
        else:
            # Vision dropout: filter coasts on IMU-driven prediction
            pass

        # Filtered estimate after correct (or just prediction if no measurement)
        est = kf.statePost
        ex, ey = int(est[0]), int(est[1])

        # Draw prediction/estimate
        cv2.circle(frame, (int(px), int(py)), 6, (0, 0, 255), 2)   # predicted
        cv2.circle(frame, (ex, ey), 6, (255, 0, 0), 2)            # estimated (fused)

        cv2.putText(frame, "White=vision meas | Red=pred | Blue=fused", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2)

        cv2.imshow("Fusion Demo", frame)
        cv2.imshow("Vision Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
