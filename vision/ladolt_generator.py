import cv2
import json
import numpy as np


def rotated_rectangle(image, center, size, angle, color):
    angle = np.rad2deg(angle)
    rect = cv2.boxPoints(((center[0], center[1]), (size[0], size[1]), angle))
    rect = np.int32(rect)
    cv2.fillConvexPoly(image, rect, color)
    return image


def generate_ladolt():
    rotations = np.random.uniform(0, 2 * np.pi, (3,))

    edge_perc = .3

    image_dim = 640
    image = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)

    center = (int(image_dim * .5), int(image_dim * .5))
    BLACK = (0, 0, 0)
    RINGS_COLOR = (128, 128, 128)

    radious = image_dim * .5
    for i, r in enumerate(rotations):
        
        cv2.circle(image, center, int(radious), RINGS_COLOR, -1)
        cv2.circle(image, center, int(radious*(1-edge_perc)), BLACK, -1)

        pos = np.array([np.cos(r), np.sin(r)]) * radious*(1-(edge_perc*.5)) + np.array([image_dim*.5, image_dim*.5])
        dims = np.array([radious * (edge_perc+.05), radious * (edge_perc+.05)])
        image = rotated_rectangle(image, pos, dims, r, BLACK)
        # cv2.circle(image, (int(pos[0]), int(pos[1])), 10, (255, 0, 0), -1)
        radious *= .6

    return image, rotations

if __name__ == "__main__":
    SAMPLE_COUNT = 1000
    rotations = []
    for i in range(SAMPLE_COUNT):
        image_path = f'../data/simple_ladolts/{i}.png'
        image, rots = generate_ladolt()
        rotations.append(rots)
        cv2.imwrite(image_path, image)
    json.dump(rotations, open("../data/simple_ladolts/rotations.json", 'w'))
