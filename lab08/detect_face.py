import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    box_scores = np.asarray(box_scores)
    if box_scores.size == 0:
        return np.empty((0, 5), dtype=box_scores.dtype if box_scores.dtype != object else float)

    boxes = box_scores[:, :4]
    scores = box_scores[:, 4]
    order = np.argsort(scores)
    if candidate_size > 0:
        order = order[-candidate_size:]

    picked = []
    while order.size:
        current = order[-1]
        picked.append(current)
        if (top_k > 0 and len(picked) >= top_k) or order.size == 1:
            break

        rest = order[:-1]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        area_current = np.maximum(0, boxes[current, 2] - boxes[current, 0]) * np.maximum(
            0, boxes[current, 3] - boxes[current, 1])
        area_rest = np.maximum(0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0, boxes[rest, 3] - boxes[rest, 1])
        union = area_current + area_rest - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        order = rest[iou <= iou_threshold]

    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    resized = image.resize((320, 240), Image.BILINEAR)
    tensor = (np.asarray(resized, dtype=np.float32) - 127.0) / 128.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options,
                                   providers=available_providers or ['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    scores = np.asarray(outputs[0]).reshape(-1, 2)[:, 1]
    boxes = np.asarray(outputs[1]).reshape(-1, 4)

    valid = scores >= conf_threshold

    candidates = np.column_stack((boxes[valid], scores[valid]))
    detections = hard_nms(candidates, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in detections:
        x1 = int(np.clip(x1 * original_width, 0, original_width - 1))
        x2 = int(np.clip(x2 * original_width, 0, original_width - 1))
        y1 = int(np.clip(y1 * original_height, 0, original_height - 1))
        y2 = int(np.clip(y2 * original_height, 0, original_height - 1))
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), f"{score:.2f}", fill="red")

    image.save(output_path)
    return detections

# Edge detection
def non_maximum_suppression(magnitude, angle):
    magnitude = np.asarray(magnitude)
    angles = (np.rad2deg(angle) + 180) % 180
    output = np.zeros(magnitude.shape, dtype=np.float32)

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            theta = angles[i, j]
            if theta < 22.5 or theta >= 157.5:
                neighbors = magnitude[i, j - 1], magnitude[i, j + 1]
            elif theta < 67.5:
                neighbors = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif theta < 112.5:
                neighbors = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                neighbors = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= neighbors[0] and magnitude[i, j] >= neighbors[1]:
                output[i, j] = magnitude[i, j]
    return output

def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong_value, weak_value = 255, 50
    result = np.zeros(np.asarray(img).shape, dtype=np.uint8)
    strong = img >= high_thresh
    weak = (img >= low_thresh) & (img < high_thresh)
    result[strong] = strong_value

    # Promote weak pixels adjacent (8-connected) to a strong edge.
    for i, j in np.argwhere(weak):
        r0, r1 = max(0, i - 1), min(img.shape[0], i + 2)
        c0, c1 = max(0, j - 1), min(img.shape[1], j + 2)
        if np.any(strong[r0:r1, c0:c1]):
            result[i, j] = strong_value
    return result

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    smoothed = gaussian_filter(gray, sigma=1.4)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(smoothed, kx)
    gy = convolve(smoothed, ky)
    magnitude = np.hypot(gx, gy)
    maximum = magnitude.max()
    if maximum > 0:
        magnitude = magnitude * (255.0 / maximum)
    angle = np.arctan2(gy, gx)

    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges.astype(np.uint8)).save(output_path)
    return edges

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")