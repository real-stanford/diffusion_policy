import cv2

def find_cameras(max_index=10):
    print("Scanning for connected cameras...\n")
    available_cameras = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera working at index {index}")
            available_cameras.append(index)
        else:
            print(f"⚠️ Camera detected at index {index}, but failed to read frame.")

        cap.release()

    if not available_cameras:
        print("\n❌ No usable cameras found.")
    else:
        print(f"\nAvailable working camera indexes: {available_cameras}")

if __name__ == "__main__":
    find_cameras()
