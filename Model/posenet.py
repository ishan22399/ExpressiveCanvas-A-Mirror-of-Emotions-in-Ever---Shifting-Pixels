import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the MoveNet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

def detect_pose(image):
    # Preprocess the image
    input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    # Run model inference
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    return keypoints

def infer_emotion_from_pose(keypoints):
    # Extract keypoints
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_elbow = keypoints[7][:2]
    right_elbow = keypoints[8][:2]
    left_wrist = keypoints[9][:2]
    right_wrist = keypoints[10][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    nose = keypoints[0][:2]

    # Calculate some distances and angles
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    shoulder_to_hip = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
    left_arm_angle = np.arctan2(left_wrist[1] - left_elbow[1], left_wrist[0] - left_elbow[0])
    right_arm_angle = np.arctan2(right_wrist[1] - right_elbow[1], right_wrist[0] - right_elbow[0])
    
    # Define heuristic rules for emotion detection
    if shoulder_to_hip < shoulder_width * 0.5:
        emotion = "Angry"
    elif left_arm_angle > np.pi / 4 and right_arm_angle > np.pi / 4:
        emotion = "Disgust"
    elif left_arm_angle < -np.pi / 4 and right_arm_angle < -np.pi / 4:
        emotion = "Fear"
    elif shoulder_width > hip_width and nose[1] < (left_hip[1] + right_hip[1]) / 2:
        emotion = "Happiness"
    elif shoulder_to_hip > shoulder_width * 1.5:
        emotion = "Sad"
    elif np.abs(left_arm_angle) < np.pi / 4 and np.abs(right_arm_angle) < np.pi / 4:
        emotion = "Surprise"
    else:
        emotion = "Neutral"

    return emotion

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (as OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        keypoints = detect_pose(rgb_frame)

        # Infer emotion from pose
        emotion = infer_emotion_from_pose(keypoints)
        
        # Display emotion on the frame
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Pose to Emotion', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
