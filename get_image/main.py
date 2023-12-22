import cv2
import os
import time

# Set the directory to save captured images
save_dir = "/Users/mariusmuresan/Desktop/get_image/with_mask"

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the image width and height
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)

# Set the number of images to capture
num_images = 40  # You can adjust this number
current_image = 0

# Start capturing images
while current_image < num_images:
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow("Capture Images", frame)

    # Delay for a few seconds to reposition before capturing the next image
    print(f"Position your head for image {current_image + 1}/{num_images}. You have 0.5 seconds.")
    time.sleep(0.5)

    # Save the captured image
    image_filename = os.path.join(save_dir, f"image_{current_image}.jpg")
    cv2.imwrite(image_filename, frame)

    current_image += 1

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
