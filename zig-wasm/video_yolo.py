#!/usr/bin/env python3
import os
import queue
import sys
import threading
import time

import av
import numpy as np
import onnxruntime as ort


INPUT_W = 640
INPUT_H = 640
QUEUE_CAPACITY = 32
MODEL_PATH = "zig-out/bin/model.onnx"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def infer_output_spec(shape, total_values):
    if len(shape) >= 3:
        d1 = int(shape[-2])
        d2 = int(shape[-1])
        if d1 > d2:
            attrs = d2
            return d1, (attrs - 4) if attrs > 4 else 80, "boxes_first"
        attrs = d1
        return d2, (attrs - 4) if attrs > 4 else 80, "attributes_first"
    return total_values // (80 + 4), 80, "boxes_first"


def get_value(tensor, boxes, attrs, box_idx, attr_idx, layout):
    if layout == "boxes_first":
        return tensor[box_idx * attrs + attr_idx]
    return tensor[attr_idx * boxes + box_idx]


def decode_v8(tensor, boxes, classes, layout, conf_threshold):
    attrs = classes + 4
    expected = boxes * attrs
    if tensor.size < expected:
        return []

    out = []
    for box_idx in range(boxes):
        cx = get_value(tensor, boxes, attrs, box_idx, 0, layout)
        cy = get_value(tensor, boxes, attrs, box_idx, 1, layout)
        w = get_value(tensor, boxes, attrs, box_idx, 2, layout)
        h = get_value(tensor, boxes, attrs, box_idx, 3, layout)

        best_class = 0
        best_score = 0.0
        for class_idx in range(classes):
            score = get_value(tensor, boxes, attrs, box_idx, class_idx + 4, layout)
            if score > best_score:
                best_score = score
                best_class = class_idx

        if best_score < conf_threshold:
            continue

        out.append(
            {
                "class_id": best_class,
                "score": float(best_score),
                "x1": float(cx - (w / 2.0)),
                "y1": float(cy - (h / 2.0)),
                "x2": float(cx + (w / 2.0)),
                "y2": float(cy + (h / 2.0)),
            }
        )
    return out


def iou(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else inter_area / union


def nms(detections, iou_threshold):
    if not detections:
        return []
    sorted_det = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    for cand in sorted_det:
        suppressed = False
        for sel in keep:
            if cand["class_id"] != sel["class_id"]:
                continue
            if iou(cand, sel) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(cand)
    return keep


def producer(video_path, stream_id, out_q):
    frame_num = 0
    prep_total_ns = 0
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frame_num += 1
                prep_start_ns = time.perf_counter_ns()
                rgb = frame.reformat(width=INPUT_W, height=INPUT_H, format="rgb24").to_ndarray()
                prep_end_ns = time.perf_counter_ns()
                prep_total_ns += prep_end_ns - prep_start_ns
                out_q.put((stream_id, frame_num, rgb), block=True)
    finally:
        out_q.put((stream_id, None, {"prep_ns": prep_total_ns, "frames": frame_num}), block=True)


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <video> <producer_count>", file=sys.stderr)
        return 2

    video_path = sys.argv[1]
    producer_count = int(sys.argv[2])
    if producer_count <= 0:
        print("producer_count must be > 0", file=sys.stderr)
        return 2

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider not available: {providers}")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CUDAExecutionProvider"],
        sess_options=ort.SessionOptions(),
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    os.makedirs("output", exist_ok=True)
    out_files = [open(f"output/stream_{i}.txt", "w", encoding="utf-8") for i in range(producer_count)]
    try:
        q = queue.Queue(maxsize=QUEUE_CAPACITY)
        threads = []
        for i in range(producer_count):
            t = threading.Thread(target=producer, args=(video_path, i, q), daemon=True)
            t.start()
            threads.append(t)

        wall_start_ns = time.perf_counter_ns()
        input_tensor = np.empty((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
        scale = 1.0 / 255.0
        done = 0
        frames = 0
        infer_total_ns = 0
        queue_wait_total_ns = 0
        post_total_ns = 0
        write_total_ns = 0
        prep_total_ns = 0
        while done < producer_count:
            q_wait_start_ns = time.perf_counter_ns()
            stream_id, frame_number, rgb = q.get(block=True)
            q_wait_end_ns = time.perf_counter_ns()
            queue_wait_total_ns += q_wait_end_ns - q_wait_start_ns
            if frame_number is None:
                if rgb is not None:
                    prep_total_ns += int(rgb.get("prep_ns", 0))
                done += 1
                continue

            np.multiply(rgb[:, :, 0], scale, out=input_tensor[0, 0], casting="unsafe")
            np.multiply(rgb[:, :, 1], scale, out=input_tensor[0, 1], casting="unsafe")
            np.multiply(rgb[:, :, 2], scale, out=input_tensor[0, 2], casting="unsafe")

            start_ns = time.perf_counter_ns()
            outputs = session.run([output_name], {input_name: input_tensor})[0]
            end_ns = time.perf_counter_ns()

            _ = outputs

            frames += 1
            infer_total_ns += end_ns - start_ns

        for t in threads:
            t.join()
        wall_end_ns = time.perf_counter_ns()

        if frames == 0 or wall_end_ns <= wall_start_ns:
            print("duration_s=0.000 frames=0 fps=0.00")
            return 0

        duration_s = (wall_end_ns - wall_start_ns) / 1_000_000_000.0
        fps = (frames * 1_000_000_000.0) / (wall_end_ns - wall_start_ns)
        print(f"duration_s={duration_s:.3f} frames={frames} fps={fps:.2f}")
        if infer_total_ns > 0:
            infer_fps = (frames * 1_000_000_000.0) / infer_total_ns
        print(
            "breakdown_s "
            f"prep={prep_total_ns / 1_000_000_000.0:.3f} "
            f"queue_wait={queue_wait_total_ns / 1_000_000_000.0:.3f} "
            f"post={post_total_ns / 1_000_000_000.0:.3f} "
            f"write={write_total_ns / 1_000_000_000.0:.3f}"
        )
        return 0
    finally:
        for f in out_files:
            f.close()


if __name__ == "__main__":
    raise SystemExit(main())
