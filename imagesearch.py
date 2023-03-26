import logging
import os
import pickle
import cv2
import numpy as np
import aiofiles
from tqdm.asyncio import tqdm_asyncio


logging.basicConfig(level=logging.INFO)

async def add(image_directory, descriptor_file):
    """
    Compute and cache ORB descriptors for new images in the specified directory
    :param image_directory: Directory containing the images
    :param descriptor_file: Output file path for storing the descriptors as a dictionary
    :return: None
    """
    # Set up the detector
    detector = cv2.ORB_create()

    # Load the previously cached descriptors from the descriptor file, if it exists
    descriptors_dict = {}
    if os.path.isfile(descriptor_file):
        try:
            async with aiofiles.open(descriptor_file, 'rb') as f:
                descriptors_dict = pickle.loads(await f.read())
        except Exception as e:
            logging.error(str(e))

    # Initialize a set to store the filenames of images that have already been processed
    processed_filenames = set(descriptors_dict.keys())

    # Loop over each image in the directory and compute its descriptors if it has not been processed before
    filenames = os.listdir(image_directory)
    new_filenames = [f for f in filenames if f not in processed_filenames]
    if not new_filenames:
        logging.info('No new images found in directory.')
        return

    logging.info(f'Processing {len(new_filenames)} new images...')
    num_images_processed = 0
    for filename in tqdm_asyncio(new_filenames, desc='Caching descriptors', unit='image'):
        # Load the stored image
        try:
            async with aiofiles.open(os.path.join(image_directory, filename), 'rb') as f:
                stored_image_bytes = await f.read()
                stored_image = cv2.imdecode(np.frombuffer(stored_image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if stored_image is None:
                    raise ValueError(f'Error: Could not load image {filename}')
        except Exception as e:
            logging.error(str(e))
            continue

        # Resize the image for faster computation
        #stored_image_resized = cv2.resize(stored_image, (0, 0), fx=0.5, fy=0.5)

        # using full resolution for more more details
        stored_image_resized = stored_image


        # Detect and compute the keypoints and descriptors for the image
        stored_kp, stored_des = detector.detectAndCompute(stored_image_resized, None)

        # Ensure that the descriptors array is continuous
        stored_des_cont = np.ascontiguousarray(stored_des)

        # Store the descriptors for the image in the dictionary
        descriptors_dict[filename] = stored_des_cont

        # Increment the number of images processed
        num_images_processed += 1

    # Save the updated dictionary to a pkl file
    try:
        async with aiofiles.open(descriptor_file, 'wb') as f:
            await f.write(pickle.dumps(descriptors_dict))
    except Exception as e:
        logging.error(str(e))
    else:
        logging.info(f'{num_images_processed} new images processed and saved successfully!')





async def search(input_image_filename, descriptor_file, n=1):
    """
    Search for the best match between an input image and a set of previously cached descriptors.
    :param input_image_filename: The filename of the input image to search for.
    :param descriptor_file: The filename of the previously cached descriptors.
    :param n: The number of top matches to return.
    :return: A list of filenames of the top matches, or an empty list if no matches are found.
    """
    # Load the descriptors from the pkl file
    try:
        async with aiofiles.open(descriptor_file, 'rb') as f:
            descriptors_dict = pickle.loads(await f.read())
    except Exception as e:
        logging.error(f'Error: Could not load descriptors from file {descriptor_file}. {str(e)}')
        return []

    # Set up the detector and matcher
    detector = cv2.ORB_create()
    matcher = cv2.FlannBasedMatcher()

    # Load the input image
    try:
        async with aiofiles.open(input_image_filename, 'rb') as f:
            input_image_bytes = await f.read()
            input_image = cv2.imdecode(np.frombuffer(input_image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if input_image is None:
                raise ValueError(f'Error: Could not load image {input_image_filename}')
    except Exception as e:
        logging.error(str(e))
        return []

    # Resize the input image for faster computation
    input_image_resized = input_image

    # Detect and compute the keypoints and descriptors for the input image
    input_kp, input_des = detector.detectAndCompute(input_image_resized, None)

    # Ensure that the descriptors array is continuous
    input_des_cont = np.ascontiguousarray(input_des, dtype=np.float32)

    # Initialize variables to track the top matches
    top_matches = []
    top_distances = []

    # Loop over each stored image and compute the distance to the input image
    for filename, stored_des_cont in descriptors_dict.items():
        # Ensure that the stored descriptors array is continuous
        stored_des_cont = np.ascontiguousarray(stored_des_cont, dtype=np.float32)

        # Match the descriptors between the input image and the stored image
        matches = matcher.match(input_des_cont, stored_des_cont)

        # Check that matches is not empty
        if len(matches) == 0:
            continue

        # Calculate the distance between the keypoints
        distance = sum(match.distance for match in matches) / len(matches)

        # Add the match to the list of top matches
        if len(top_matches) < n:
            top_matches.append(filename)
            top_distances.append(distance)
        elif distance < max(top_distances):
            index = top_distances.index(max(top_distances))
            top_matches[index] = filename
            top_distances[index] = distance

    # Sort the top matches by distance
    if top_matches:
        top_matches_sorted = [match for _, match in sorted(zip(top_distances, top_matches))]
        return top_matches_sorted
    else:
        return []
