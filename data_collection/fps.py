import cv2

# Replace with your actual video file path
video_path = "data/video/human_demo_11/camera_0.mp4.mp4"
# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if opened successfully
if not cap.isOpened():
    print(f"❌ Error opening video file: {video_path}")
    exit()

# Read and display frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("🔚 End of video or can't read the frame.")
        break

    cv2.imshow("Video Playback", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
