import cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO("models/best.pt")

images = Path("test/images")
detection_output = Path("output/detections")
counting_output = Path("output/counting")
detection_output.mkdir(parents=True, exist_ok=True)
counting_output.mkdir(parents=True, exist_ok=True)

count_line_y = 300
line_color = (0, 0, 255)
line_thickness = 2 

count_dict = {}
counted_centers = []

for img_path in images.glob("*.jpg"):
    print(f"Processing image: {img_path.name}")
    
    img = cv2.imread(str(img_path))
    results = model(img_path)  # Run detection
    print(results)

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            id = int(box.cls)
            name = names[id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf)

            # Calculate center of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Count object if it crosses the line
            if center_y > count_line_y:
                object_id = (id, center_x, center_y)
                if object_id not in counted_centers:
                    counted_centers.append(object_id)
                    count_dict[name] = count_dict.get(name, 0) + 1

            # Draw bounding box, center, and label
            cv2.rectangle(img, (x1, y1), (x2, y2), line_color, line_thickness)
            cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)

        # Draw the horizontal counting line
        cv2.line(img, (0, count_line_y), (img.shape[1], count_line_y), line_color, line_thickness)

        # === Display the image ===
        cv2.imshow("Detected Objects", img)
        key = cv2.waitKey(0)  # Wait for a key press to proceed
        if key == ord('q'):
            break
# === Save outputs ===
        cv2.imwrite(str(detection_output / img_path.name), result.plot())   # with YOLO visuals
        cv2.imwrite(str(counting_output / img_path.name), img)              # with custom overlays

cv2.destroyAllWindows()

# Print final counts
print("\nFinal Counts:")
for obj, count in count_dict.items():
    print(f"{obj}: {count}")
