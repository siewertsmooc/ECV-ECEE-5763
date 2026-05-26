import cv2
import numpy as np

def fuse_two_rgb_images(imgA, imgB, ratio=0.75, reproj_thresh=4.0):
    """
    Fuse two RGB images by:
      1) Detecting/matching features
      2) Estimating a homography
      3) Warping one image into the other's frame
      4) Blending overlapping regions
    """
    # Use ORB (fast, free). SIFT works too if available in your build.
    orb = cv2.ORB_create(nfeatures=3000)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    kpsA, desA = orb.detectAndCompute(grayA, None)
    kpsB, desB = orb.detectAndCompute(grayB, None)
    if desA is None or desB is None or len(kpsA) < 10 or len(kpsB) < 10:
        raise RuntimeError("Not enough features to fuse these images.")

    # KNN match + Lowe ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desA, desB, k=2)

    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 12:
        raise RuntimeError(f"Not enough good matches ({len(good)}) to estimate homography.")

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reproj_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    # Warp B into A's coordinate system with a big enough canvas
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]

    cornersB = np.float32([[0,0], [wB,0], [wB,hB], [0,hB]]).reshape(-1,1,2)
    warped_cornersB = cv2.perspectiveTransform(cornersB, H)

    cornersA = np.float32([[0,0], [wA,0], [wA,hA], [0,hA]]).reshape(-1,1,2)

    all_corners = np.vstack((cornersA, warped_cornersB))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)

    out_w = xmax - xmin
    out_h = ymax - ymin

    warpedB = cv2.warpPerspective(imgB, T @ H, (out_w, out_h))
    canvasA = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvasA[ty:ty+hA, tx:tx+wA] = imgA

    # Simple blend: feather overlap via masks
    maskA = (canvasA.sum(axis=2) > 0).astype(np.float32)
    maskB = (warpedB.sum(axis=2) > 0).astype(np.float32)

    overlap = (maskA > 0) & (maskB > 0)

    # Distance transform feathering for nicer seams
    invA = (1 - maskA).astype(np.uint8)
    invB = (1 - maskB).astype(np.uint8)
    distA = cv2.distanceTransform(invA, cv2.DIST_L2, 3).astype(np.float32)
    distB = cv2.distanceTransform(invB, cv2.DIST_L2, 3).astype(np.float32)

    # Where both exist, weight by distance-to-edge (bigger distance => more confident)
    wA = distA / (distA + distB + 1e-6)
    wB = distB / (distA + distB + 1e-6)

    # Where only one exists, give it full weight
    wA[maskA > 0] = np.maximum(wA[maskA > 0], 0.0)
    wB[maskB > 0] = np.maximum(wB[maskB > 0], 0.0)
    wA[(maskA > 0) & (maskB == 0)] = 1.0
    wB[(maskB > 0) & (maskA == 0)] = 1.0

    fused = (canvasA.astype(np.float32) * wA[..., None] +
             warpedB.astype(np.float32) * wB[..., None])

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused

if __name__ == "__main__":
    # Replace with your own two-camera frames or two images from disk:
    imgA = cv2.imread("camA.png")
    imgB = cv2.imread("camB.png")
    if imgA is None or imgB is None:
        raise SystemExit("Put camA.png and camB.png next to this script (or change paths).")

    fused = fuse_two_rgb_images(imgA, imgB)

    cv2.imshow("A", imgA)
    cv2.imshow("B", imgB)
    cv2.imshow("Fused (mosaic)", fused)
    cv2.waitKey(0)
