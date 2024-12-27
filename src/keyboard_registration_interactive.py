import cv2
import numpy as np

# Initialize a list to store the points
points = []

def draw_points_and_ask_for_confirmation(image):
    temp_image = image.copy()
    for point in points:
        cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
    cv2.imshow('image', temp_image)

    print("Press 'c' to confirm points, 'r' to reset all points.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):  # Confirm points
            print("Pressed 'c'")
            return True
        elif key == ord('r'):  # Reset points
            print("Pressed 'r'")
            points.clear()
            cv2.imshow('image', image)
            return False

def click_event(event, x, y, flags, params):
    # mouse callback function
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) <= 4:
            points.append((x, y))

        if len(points) == 4:
            cv2.imshow('image', img)  # Refresh image to show all points
            if not draw_points_and_ask_for_confirmation(img):
                click_event(event, x, y, flags, params)  # Recursively call to handle new points

import numpy as np
import cv2

def order_points(quadrilateral_points):
    # Initialize a list of coordinates that will be ordered
    # in a specific manner: top-left, top-right, bottom-right, bottom-left
    ordered_rect = np.zeros((4, 2), dtype="float32")

    # Calculate the sum and difference of the points
    sum_points = quadrilateral_points.sum(axis=1)
    diff_points = np.diff(quadrilateral_points, axis=1)

    # Assigning corners based on the sum and difference
    ordered_rect[0] = quadrilateral_points[np.argmin(sum_points)]  # Top-left
    ordered_rect[2] = quadrilateral_points[np.argmax(sum_points)]  # Bottom-right
    ordered_rect[1] = quadrilateral_points[np.argmin(diff_points)]  # Top-right
    ordered_rect[3] = quadrilateral_points[np.argmax(diff_points)]  # Bottom-left

    return ordered_rect

def warp_perspective(image, points):
    # Ensure points are in a float32 numpy array
    rect = order_points(np.array(points, dtype="float32"))
    (top_left, top_right, bottom_right, bottom_left) = rect

    # Compute widths and heights of the image
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    maxWidth = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(top_left - bottom_left)
    height_right = np.linalg.norm(top_right - bottom_right)
    maxHeight = max(int(height_left), int(height_right))

    # Destination points for the "birds eye view"
    destination_points = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    transform_matrix = cv2.getPerspectiveTransform(rect, destination_points)
    warped_image = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

    return warped_image

def get_perspective_transformed_image(image):
    global img
    img = image.copy()
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('z') and points:  # Undo last point
            points.pop()
            img = image.copy()
            for point in points:
                cv2.circle(img, point, 5, (0, 255, 0), -1)
            cv2.imshow('image', img)
        elif key == ord('r'):  # Reset all points
            points.clear()
            img = image.copy()
            cv2.imshow('image', img)
        elif key == ord('q') or len(points) == 4:  # Quit or move on after confirmation
            break

    if len(points) != 4:
        print("Operation not confirmed. Exiting...")
        return None

    # Assuming points are confirmed and correctly ordered as needed
    result = warp_perspective(img, points)

    return result

# Load your image
image_path = '../../resources/images/piano_10.png'  # Change this to the path of your image
image = cv2.imread(image_path)
if image is None:
    print("Image not found:", image_path)
else:
    warped_image = get_perspective_transformed_image(image)

    if warped_image is not None:
        cv2.imshow("Warped Image", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
