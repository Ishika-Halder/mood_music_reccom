import cv2
import joblib
from fer import FER
import random
import webbrowser
import os
MODEL_PATH = "models/song_index.pkl"

def main():
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")

    # Load mood→songs index
    mood_index = joblib.load(MODEL_PATH)
    print(f"[OK] Loaded model with {len(mood_index)} moods.")

    # Initialize webcam
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0) 

    print("Press 'q' to quit the camera window after mood detection.")

    detected_mood = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect emotions
        result = detector.detect_emotions(frame)

        if result:
            # Pick the first face detected
            emotions = result[0]["emotions"]
            detected_mood = max(emotions, key=emotions.get)

            # Show the mood on the camera feed
            cv2.putText(frame, f"Mood: {detected_mood}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Mood Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if detected_mood is None:
        print("No mood detected.")
        return

    print(f"Detected Mood: {detected_mood}")

    # Recommend songs for the detected mood
    if detected_mood in mood_index:
        songs = mood_index[detected_mood]
        print(f"Found {len(songs)} songs for mood '{detected_mood}':")

        for song in songs:
            print(f"- {song['title']} by {song['artist']} ({song['url']})")

        # Play a random song
        choice = random.choice(songs)
        print(f"\n🎵 Playing: {choice['title']} by {choice['artist']}")
        webbrowser.open(choice["url"])
    else:
        print(f"No songs available for mood '{detected_mood}'.")

if __name__ == "__main__":
    main()
