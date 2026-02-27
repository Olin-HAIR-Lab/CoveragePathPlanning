import cv2
import numpy as np
from shapely.geometry import Polygon

def image_to_polygon_points(image_path, num_points=50, scale_to_range=(-10, 10)):
    """
    Takes a silhouette image and returns outline points as a Shapely Polygon.

    Args:
        image_path: Path to the image file
        num_points: Number of points to sample along the contour
        scale_to_range: Tuple (min, max) to normalize coordinates into

    Returns:
        Polygon object and list of (x, y) tuples
    """
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"image_to_polygon_points: could not read '{image_path}'; check the path and file format")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image.  We usually want the object to be
    # white on black, but some input images are opposite.  Try THRESH_BINARY_INV
    # first and if the largest contour covers essentially the whole image, redo
    # without inversion.
    h, w = gray.shape

    def find_contours(binary_img):
        return cv2.findContours(binary_img, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0]

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours = find_contours(binary)

    if contours:
        largest_area = max(cv2.contourArea(c) for c in contours)
        if largest_area >= 0.99 * w * h:
            # probably inverted: shape is white on dark; recompute without inv
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours = find_contours(binary)

    if not contours:
        # last resort: try adaptive thresholding in case of poor contrast
        adaptive = cv2.adaptiveThreshold(gray, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        contours = find_contours(adaptive)
        if not contours:
            raise ValueError(
                "No contours found in image; check that the file contains a "
                "distinct silhouette or try a different thresholding method.")

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Resample contour to desired number of points
    contour = contour.squeeze()  # Remove extra dimension: (N, 1, 2) -> (N, 2)

    # Evenly sample num_points from the contour
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
    sampled = contour[indices]

    # Get image dimensions for normalization
    h, w = gray.shape
    min_val, max_val = scale_to_range

    # Normalize coordinates to target range
    # X: map [0, w] -> [min_val, max_val]
    # Y: map [0, h] -> [max_val, min_val] (flip Y so top = higher value)
    def normalize_x(x):
        return round(min_val + (x / w) * (max_val - min_val), 2)

    def normalize_y(y):
        # Flip Y axis (image Y goes down, but we want Y going up)
        return round(max_val - (y / h) * (max_val - min_val), 2)

    points = [(normalize_x(x), normalize_y(y)) for x, y in sampled]

    # Close the polygon by repeating the first point
    points.append(points[0])

    poly = Polygon(points)
    return poly, points


def print_polygon_code(points):
    """Prints the points as Python code like in the example."""
    print("poly = Polygon([")
    for pt in points:
        print(f"    {pt},")
    print("])")


# Example usage
if __name__ == "__main__":
    image_path = "tree.png"  # Replace with your image path

    poly, points = image_to_polygon_points(
        image_path,
        num_points=30,         # Adjust for more/fewer points
        scale_to_range=(-10, 10)  # Adjust coordinate range
    )

    print_polygon_code(points)
    print(f"\nPolygon area: {poly.area:.2f}")
    print(f"Polygon is valid: {poly.is_valid}")
