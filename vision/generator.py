import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from noise import snoise2

from ladolt_generator import generate_ladolt


SAMPLE_COUNT = 10_000
dataset_dir = "../data/dogs"
room_images_dirs = os.listdir(dataset_dir)


def add_simplex_noise(image: np.ndarray, alpha: float):
    height, width, channels = image.shape

    scale = 100.0
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0

    noise_array = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            noise_value = snoise2(x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            noise_array[y][x] = (noise_value + 1) * 127.5

    noise_array = noise_array.astype(np.uint8)

    noise_array_colored = cv2.merge([noise_array, noise_array, noise_array])

    noisy_image = cv2.addWeighted(image, 1 - alpha, noise_array_colored, alpha, 0)

    return noisy_image



def perspective_transform(image: np.ndarray, dst_points: np.ndarray) -> np.ndarray:

    height, width = image.shape[:2]

    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    alpha_channel = np.ones((height, width), dtype=np.uint8) * 255
    mask = np.any(transformed_image != 0, axis=2)
    alpha_channel[~mask] = 0
    
    transformed_image = add_simplex_noise(transformed_image, np.random.uniform(low=.2, high=.7))

    b, g, r = cv2.split(transformed_image)
    rgba_image = cv2.merge((b, g, r, alpha_channel))
    
    return rgba_image


def overlay_images(background: np.ndarray, foreground: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    
    h, w = foreground.shape[:2]
    if y_offset + h > background.shape[0] or x_offset + w > background.shape[1]:
        raise ValueError("Foreground image exceeds background boundaries.")
    b, g, r, a = cv2.split(foreground)
    alpha_mask = a / 255.0

    for c in range(3):
        background[y_offset:y_offset + h, x_offset:x_offset + w, c] = (
            (1 - alpha_mask) * background[y_offset:y_offset + h, x_offset:x_offset + w, c] +
            alpha_mask * foreground[:, :, c]
        )

    return background


if __name__ == "__main__":
    rots = []
    if not os.path.exists(f"./datasets/ladolt_dataset"):
        os.makedirs(f"./datasets/ladolt_dataset/images/val/")
        os.makedirs(f"./datasets/ladolt_dataset/images/train/")
        os.makedirs(f"./datasets/ladolt_dataset/labels/val/")
        os.makedirs(f"./datasets/ladolt_dataset/labels/train/")
        os.makedirs(f"./datasets/cropped/")
        
    for i in tqdm(range(SAMPLE_COUNT)):
        category_folder = np.random.choice(room_images_dirs)
        background_path = np.random.choice(os.listdir(f"{dataset_dir}/{category_folder}"))
        background_path = f"{dataset_dir}/{category_folder}/{background_path}"
        
        background = cv2.imread(background_path)
        foreground, rotations = generate_ladolt()

        if background is None or foreground is None:
            print("Could not load ", background_path)
            continue

        bh, bw = background.shape[:2]
        fore_size = np.random.uniform(low=.07, high=.9, size=(2,))
        if bw < bh:
            fore_size[1] = fore_size[0]*bw/bh
        else:
            fore_size[0] = fore_size[1]*bh/bw
        reshaped = (int(fore_size[0]*bw), int(fore_size[1]*bh))
        foreground = cv2.resize(foreground, reshaped)

        height, width = foreground.shape[:2]

        value = np.random.randint(0, 4)

        scaling = np.random.uniform(low=.0, high=.4)

        shift = 1-np.random.uniform(low=-scaling, high=scaling)

        if value == 0:
            dst_points = np.float32([[0, 0],
                                     [width, 0],
                                     [width * (1-scaling), height],
                                     [width * scaling, height]])
        elif value == 1:
            dst_points = np.float32([[width * scaling - shift, 0],
                                     [width * (1-scaling) - shift, 0],
                                     [width, height],
                                     [0, height]])
        elif value == 2:
            dst_points = np.float32([[0, height * scaling - shift],
                                     [width, 0],
                                     [width, height],
                                     [0, height * (1-scaling) - shift]])
        elif value == 3:
            dst_points = np.float32([[0, 0],
                                     [width, height * scaling - shift],
                                     [width, height * (1-scaling) - shift],
                                     [0, height]])

        foreground_transformed = perspective_transform(foreground, dst_points)

        x_offset = int(np.random.uniform(low=.0, high=1.-fore_size[0]) * bw)
        y_offset = int(np.random.uniform(low=.0, high=1.-fore_size[1]) * bh)

        result_image = overlay_images(background, foreground_transformed, x_offset, y_offset)

        if i/SAMPLE_COUNT > .8:
            cv2.imwrite(f'./datasets/ladolt_dataset/images/val/img{i+1}.jpg', result_image)
            label_file = open(f'./datasets/ladolt_dataset/labels/val/img{i+1}.txt', 'w')
            label_file.write(f"0 {x_offset/bw} {y_offset/bh} {width/bw} {height/bh}")
            json.dump(rotations.tolist(), open(f"./datasets/ladolt_dataset/labels/val/img{i+1}.json", 'w'))
        else:
            cv2.imwrite(f'./datasets/ladolt_dataset/images/train/img{i+1}.jpg', result_image)
            label_file = open(f'./datasets/ladolt_dataset/labels/train/img{i+1}.txt', 'w')
            label_file.write(f"0 {x_offset/bw} {y_offset/bh} {width/bw} {height/bh}")
            json.dump(rotations.tolist(), open(f"./datasets/ladolt_dataset/labels/train/img{i+1}.json", 'w'))

    #     cv2.imwrite(f'./datasets/cropped/{i}.png', result_image[y_offset:y_offset+height, x_offset:x_offset+width])
    #     rots.append(rotations[0])
    # json.dump(rots, open(f"./datasets/cropped/rotations.json", 'w'))
