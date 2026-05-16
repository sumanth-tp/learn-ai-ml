---
title: OpenCV Computer Vision Master Cheatsheet
sidebar_position: 20
---

# OpenCV Computer Vision Master Cheatsheet

## Image I/O and basics

| Method | Description | Code example |
|---|---|---|
| `cv2.imread()` | `cv2.imread(filename, flags=cv2.IMREAD_COLOR)` reads image as BGR NumPy array. | `import cv2`<br/>`img = cv2.imread("image.jpg")`<br/>`print(img.shape)` |
| `cv2.imwrite()` | Writes image to disk. | `cv2.imwrite("output.png", img)` |
| BGR to RGB | OpenCV uses BGR; matplotlib expects RGB. | `rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Resize | `cv2.resize(src, dsize, interpolation=None)` changes image size. | `resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)` |
| Crop | NumPy slicing crops images. | `crop = img[y1:y2, x1:x2]` |
| Draw | Draw boxes, text, circles, and lines. | `cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)`<br/>`cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)` |

## Color, filtering, and transforms

| Method | Description | Code example |
|---|---|---|
| Grayscale | Converts color image to one channel. | `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| HSV | Useful for color thresholding. | `hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` |
| Gaussian blur | Smooths noise before edge detection or thresholding. | `blur = cv2.GaussianBlur(gray, (5, 5), 0)` |
| Threshold | Converts grayscale to binary image. | `_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)` |
| Adaptive threshold | Handles uneven lighting. | `binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)` |
| Morphology | Erode/dilate/open/close masks. | `kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))`<br/>`clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)` |
| Affine transform | Rotate or translate images. | `matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)`<br/>`rotated = cv2.warpAffine(img, matrix, (w, h))` |

## Edges, contours, and features

| Method | Description | Code example |
|---|---|---|
| Canny edge | Detects edges from grayscale image. | `edges = cv2.Canny(gray, threshold1=100, threshold2=200)` |
| Find contours | Extracts connected boundaries from binary masks. | `contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` |
| Bounding box | Rectangle around contour. | `for contour in contours:`<br/>`    x, y, w, h = cv2.boundingRect(contour)`<br/>`    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)` |
| Contour area | Filters small noisy detections. | `large = [c for c in contours if cv2.contourArea(c) > 500]` |
| Hough lines | Detects straight lines. | `lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)` |
| ORB features | Fast keypoints and descriptors. | `orb = cv2.ORB_create()`<br/>`keypoints, descriptors = orb.detectAndCompute(gray, None)` |

## Video and PyTorch integration

| Method | Description | Code example |
|---|---|---|
| `VideoCapture` | Reads frames from camera or video file. | `cap = cv2.VideoCapture("video.mp4")`<br/>`ok, frame = cap.read()` |
| Frame loop | Process video frame by frame. | `while cap.isOpened():`<br/>`    ok, frame = cap.read()`<br/>`    if not ok:`<br/>`        break`<br/>`    process(frame)` |
| `VideoWriter` | Writes processed video. | `writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))`<br/>`writer.write(frame)` |
| NumPy to Torch | Convert image to channel-first normalized tensor. | `rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`<br/>`tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0` |
| Torch to OpenCV | Convert tensor back to BGR image. | `arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")`<br/>`bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Read and display with matplotlib | Convert BGR to RGB first. | `img = cv2.imread("image.jpg")`<br/>`plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))`<br/>`plt.axis("off")` |
| Color mask | Segment by HSV range. | `hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`<br/>`mask = cv2.inRange(hsv, lower, upper)` |
| Deskew image | Rotate based on detected angle. | `coords = np.column_stack(np.where(binary > 0))`<br/>`angle = cv2.minAreaRect(coords)[-1]` |
| OCR preprocessing | Grayscale, blur, threshold, morphology. | `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`<br/>`blur = cv2.GaussianBlur(gray, (3, 3), 0)`<br/>`_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` |
| Object crop from box | Crop detection result. | `crop = img[y1:y2, x1:x2]`<br/>`cv2.imwrite("object.jpg", crop)` |
| Batch resize dataset | Prepare image folder. | `for path in Path("images").glob("*.jpg"):`<br/>`    img = cv2.imread(str(path))`<br/>`    cv2.imwrite(str(out_dir / path.name), cv2.resize(img, (224, 224)))` |
| Webcam preview | Live camera loop. | `cap = cv2.VideoCapture(0)`<br/>`while True:`<br/>`    ok, frame = cap.read()`<br/>`    if not ok: break`<br/>`    cv2.imshow("frame", frame)` |
| Release resources | Always release video handles and windows. | `cap.release()`<br/>`writer.release()`<br/>`cv2.destroyAllWindows()` |

## Senior CV preprocessing

| Method | Description | Code example |
|---|---|---|
| Letterbox resize | Resize while preserving aspect ratio for detection models. | `scale = min(new_w / w, new_h / h)`<br/>`resized = cv2.resize(img, (int(w * scale), int(h * scale)))`<br/>`canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)` |
| Normalize for model | Apply channel order, scale, mean, and std exactly as training. | `rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`<br/>`x = rgb.astype("float32") / 255.0`<br/>`x = (x - mean) / std` |
| Deterministic augmentation | Seed random transforms when debugging training issues. | `rng = np.random.default_rng(seed)`<br/>`if rng.random() < 0.5:`<br/>`    img = cv2.flip(img, 1)` |
| Camera calibration | Estimate intrinsic matrix and distortion coefficients. | `ok, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)` |
| Undistort | Correct lens distortion before measurement tasks. | `undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)` |
| Perspective transform | Rectify planar surfaces such as documents. | `M = cv2.getPerspectiveTransform(src_pts, dst_pts)`<br/>`warped = cv2.warpPerspective(img, M, (width, height))` |
| Connected components | Label binary mask regions. | `num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)` |
| Non-max suppression | Remove duplicate detection boxes. | `indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.5)` |

## Production CV systems

| Method | Description | Code example |
|---|---|---|
| Frame sampling | Process every Nth frame to control cost. | `if frame_idx % sample_rate != 0:`<br/>`    continue` |
| Timestamp preservation | Keep frame timestamps for audit and alignment. | `timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)` |
| Decode failure handling | Skip corrupt frames and count them. | `ok, frame = cap.read()`<br/>`if not ok:`<br/>`    decode_errors += 1` |
| Batch inference | Stack preprocessed frames for GPU efficiency. | `batch = np.stack([preprocess(frame) for frame in frames])`<br/>`preds = model(torch.from_numpy(batch).to(device))` |
| Annotation audit | Save images with predictions for manual review. | `cv2.imwrite(f"audit/{image_id}.jpg", annotated)` |
| Coordinate scaling | Map model input boxes back to original image size. | `x1 = int(x1_model / scale)`<br/>`y1 = int(y1_model / scale)` |
| Latency budget | Measure decode, preprocess, inference, postprocess separately. | `timings = {"decode": t1 - t0, "pre": t2 - t1, "infer": t3 - t2}` |
| Privacy blur | Blur faces or plates before storage. | `roi = img[y1:y2, x1:x2]`<br/>`img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)` |
