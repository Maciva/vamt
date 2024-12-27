import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import argparse
import time

parser = argparse.ArgumentParser(description="Visual automatic music transcription")
parser.add_argument('--input', '-i', required=True, type=str, help='input file path')
parser.add_argument('--threshold_w', default=4 ,type=int, help='Thresholdvalue of white keys to detect a keypress. Can be finetuned according to optimize precision/recall')
parser.add_argument('--threshold_b', default=5 ,type=int, help='Thresholdvalue of black keys to detect a keypress, can be finetuned according to optimize precision/recall')
parser.add_argument('--show_hist_metrics', default=False , action='store_true', help='Plot histogram of the metric which get thresholded for keypress detection.')
parser.add_argument('--hand_mask', default=False , action='store_true', help='Visualize masked hands.')
parser.add_argument('--difference_images', default=False, action='store_true', help='Visualize difference images between frames and background image.')

args = parser.parse_args()


def find_outlier_indices(data):
	"""finds outliers within the list of data

	Args:
		data: 1D array of data to find outliers in

	Returns:
		list: array of indices which are considered outliers in the array
	"""
	Q1 = np.percentile(data, 25)
	Q3 = np.percentile(data, 75)
	IQR = Q3 - Q1
	multiplier = 1.5
	return np.where((data < (Q1 - multiplier * IQR)) | (data > (Q3 + multiplier * IQR)))[0]

def normalize_image(image):
	min_val = np.min(image)
	max_val = np.max(image)
	normalized_image = (image - min_val) / (max_val - min_val)
	return (normalized_image * 255).astype(np.uint8)

def add_without_blend(img1, img2):
	return np.where(np.any(img2>0,axis=-1,keepdims=True), img2, img1)   

def extract_mean_black_key_distances(data):
	"""extracts the mean distance within the black_key_distances, where two distance clusters are considered (near distance and far distance)

	Args:
		data: black_key_distances

	Returns:
		near_distance: mean near distance between black keys
		far_distance: mean far distance between black keys
	"""
	x = np.array(data)
	km = KMeans(2)
	y = x.reshape(-1 ,1)
	km.fit(y)
	far_index = km.cluster_centers_.argmax()
	near_index = 1 - far_index
	return km.cluster_centers_[near_index][0], km.cluster_centers_[far_index][0]

def get_w_b_keys_masks(image_grayscale):
	"""applies piecewise otsus threhsolding on a grayscale image to extract the black and white keys. Closing and opening morphologies are applied to remove noise and holes

	Args:
		image_grayscale

	Returns:
		white_key_mask,
		black_key_mask
	"""
	otsu_mask = piecewise_otsu_thresholding(image_grayscale)
	kernel_size = 3
	kernel = np.ones((kernel_size*4, kernel_size*2), np.uint8)
	closed_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

	white_key_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
	black_key_mask = (255 - white_key_mask)
	return white_key_mask, black_key_mask


def extract_black_key_info(black_key_mask):
	"""
	applies connected components algorithm on the black key mask. Removing outliers by area ensures that only the black keys are conisdered
	furthermore, differences to preceeding keys are stored, to later apply k means to it, to extract the mean black key distance
	Args:
		black_key_mask: mask of black keys ranging from 0 to 255

	Returns:
		black_keys_bb: all bounding boxes for black keys
		delta_x_to_previous: distances between black keys
	"""
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_key_mask, connectivity=4)
	delta_x_to_previous = []
	outliers_by_area = find_outlier_indices(stats[:, cv2.CC_STAT_AREA])
	modified_range = np.delete(range(0, num_labels), outliers_by_area)
	black_keys_bb = []

	for i in modified_range:
		if i in outliers_by_area:
			continue
		x = stats[i, cv2.CC_STAT_LEFT]
		y = stats[i, cv2.CC_STAT_TOP]
		w = stats[i, cv2.CC_STAT_WIDTH]
		h = stats[i, cv2.CC_STAT_HEIGHT]
		(cX, cY) = centroids[i]
		black_keys_bb.append((x,y,x+w,y+h))
		
		if i != modified_range[0]:
			(xP, yP) = centroids[i-1]
			delta_x_to_previous.append(cX - xP)
	return black_keys_bb, delta_x_to_previous

def apply_sobel(image_grayscale, mask):
	"""applies sobelfilter onto a grayscale image, whilst ignoring areas within a given mask.

	Args:
		image_grayscale
		mask: areas to ignore

	Returns:
		resulting image after filtering, which is masked by given mask
	"""
	kernel_size = (15, 10)
	kernel = np.ones(kernel_size, np.uint8)

	sobelx = cv2.Sobel(image_grayscale, cv2.CV_64F, 1, 0, ksize=3)
	sobelx_magnitude = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))
	extended_mask = cv2.erode(mask, kernel, iterations=1)
	return cv2.bitwise_and(sobelx_magnitude, sobelx_magnitude, mask=extended_mask)

def piecewise_otsu_thresholding(input):
	"""applies otsus thresholding onto given input image

	Args:
		input: input image

	Returns:
		resulting mask of otus thresholding
	"""
	h, w = input.shape

	chunk_size = (int(w / 5), int(h / 5))
	mask = np.zeros_like(input, dtype=np.uint8)

	for y in range(0, h, chunk_size[1]):
		for x in range(0, w, chunk_size[0]):
			roi = input[y:y+chunk_size[1], x:x+chunk_size[0]]
			
			_, chunk_mask = cv2.threshold(roi, 0, 255,cv2.THRESH_OTSU)
			
			mask[y:y+chunk_size[1], x:x+chunk_size[0]] = chunk_mask
	return mask

def generate_white_key_bb(seperators, image_height):
	"""generates white key bounding boxes by a given list of line seperators between the white keys

	Args:
		seperators: list of lineseperators of white keys
		image_height: height of image

	Returns:
		list of white key boundingboxes
	"""
	result = []  
	for i in range(1, len(seperators)):
		result.append([seperators[i-1], 0, seperators[i], image_height])
	return result

def extract_white_key_bb(white_key_seperators_mask, black_key_near_distance):
	"""applies the assumption of vertical lines, to reduce the problem to a 1Dimensional problem, by reducing the y-axis to the sum of all y-elements.
		The resulting peaks in the 1D distribution can be assumed to be the white key seperators, whilst complying with the black_key_near_distance

	Args:
		white_key_seperators_mask: mask of white key seperators previously applied with threhsolding
		black_key_near_distance: distance between two black keys which are nearby

	Returns:
		white_key_bb: list of white bounding boxes
	"""
	vertical_sum = np.sum(white_key_seperators_mask, axis=0)
	vertical_sum_normalized = vertical_sum / np.max(vertical_sum) * 255
	white_key_seperators, _ = find_peaks(vertical_sum_normalized, distance= black_key_near_distance * 0.5)
	white_key_seperators = np.append(white_key_seperators, white_key_seperators_mask.shape[1])
	white_key_seperators = np.insert(white_key_seperators, 0, 0)
	return generate_white_key_bb(white_key_seperators, white_key_seperators_mask.shape[0])

def generate_keys_dict(black_keys_bb, white_keys_bb):
	"""merges two lists of black and white keys, extending the data strcuture by type and minimum x component.

	Args:
		black_keys_bb
		white_keys_bb
	Returns:
		sorted_keys: merged and sorted list by min x of black and white keys
		black: dict of black key bbs with given indices
		white: dict of white key bbs with given indices
	"""
	merged_list = []
	for black_key_bb in black_keys_bb:
		merged_list.append( {'x': black_key_bb[0], 'type': 'black', 'data': black_key_bb} )
	for white_key_bb in white_keys_bb:
		merged_list.append( {'x': white_key_bb[0], 'type': 'white', 'data': white_key_bb})
		
	sorted_keys = sorted(merged_list, key=sort_key)
	black, white = seperate_sorted_keys(sorted_keys)
	return sorted_keys, black, white

def sort_key(key_tuple):
	return key_tuple['x']

def seperate_sorted_keys(sorted_list):
	"""seperates sorted list of keys by type

	Args:
		sorted_list: list of black and white keys

	Returns:
		black: dict of black keys,
		white: dict of white keys
	"""
	black = {}
	white = {}
	for i in range(len(sorted_list)):
		element = sorted_list[i]
		if element['type'] == 'white':
			white[i] = element['data']
		else:
			black[i] = element['data']
	return black, white

def find_first_fis_index(black_keys):
	"""finds the first fis key of given bounding boxes, where two proceeding black keys have to be two half steps apart

	Args:
		black_keys: dict of black keys with index => bounding box

	Returns:
		index of first fis key
	"""
	for i in black_keys:
		if (i+2) in black_keys and (i+4) in black_keys:
			return i
	return None


def generate_piano_keys():
	"""generates a list of piano labels

	Returns:
		keys: list of piano labels
	"""
	notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	keys = []
	
	for octave in range(0, 9):
		for note in notes:
			keys.append(f"{note}{octave}")
	return keys


def generate_labels(piano_labels, first_fis_reference):
	"""generates the labeling of indexes and piano labels, given the index of the first fis key

	Args:
		piano_labels: list of labels on the keyboard
		first_fis_reference: index of the first fis key to refference

	Returns:
		dict: dict of index => label
	"""
	result = {}
	FIRST_FIS_REFERENCE_GT = 18
	offset = FIRST_FIS_REFERENCE_GT - first_fis_reference
	if offset < 0:
		offset += 12
	for i in range(offset, len(piano_labels)):
		result[i - offset] = piano_labels[i]
	return result    

def generate_overlay_segmentation(image, mask, labels, black_key_near_distance):
	"""generates an overlay for the segmentation of keys

	Args:
		image: input image
		mask: segmentation mask of all keys
		labels: dict of all labes
		black_key_near_distance: mean distance of black keys which are nearby

	Returns:
		overlay image containing key bounding boxes and their respective labels
	"""
	image_with_edges = image.copy()
	
	for label in np.unique(mask):
		binary_mask = np.zeros(mask.shape, dtype=np.uint8)
		binary_mask[mask == label] = 255
		
		contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		cv2.drawContours(image_with_edges, contours, -1, (0, 255, 0), 1)
		
		for contour in contours:
			M = cv2.moments(contour)
			if M["m00"] != 0:
				cx = int(M["m10"] / M["m00"])
				cy = int(M["m01"] / M["m00"])
				cv2.putText(image_with_edges, f"{labels[label]}", (cx - int(black_key_near_distance * 0.2), cy + int(black_key_near_distance)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	
	return image_with_edges

def process_background_image(image_bgr):
	"""processes and extracts all required background information

	Args:
		image_bgr 3d-array: input image to process

	Returns:
		keys_overlay: an overlay image, displaying the key segmentation aswell as their labels
		keys_dict: dictionary of indexes to their mapped bb
		labels: dictionary of indexes mapped to the key label
		segmentation: segmentation mask of all keys listed in keys_dict
	"""
	image_grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	# extract white and black key masks
	white_key_mask, black_key_mask = get_w_b_keys_masks(image_grayscale)
	# exctract black key boundingboxes and their distance from one another
	black_keys_bb, delta_x_to_previous = extract_black_key_info(black_key_mask)
	# find mean near and mean far distance of black keys from one another
	black_key_near_distance, black_key_far_distance = extract_mean_black_key_distances(delta_x_to_previous)

	# apply 1d sobelfilter to detect vertical lines
	white_keys_vertical_edges = apply_sobel(image_grayscale, white_key_mask)
	# apply a piecewise otsu thresholding to seperate edges from noise aswell as counteract brightness changes over the image
	white_key_seperators_mask = piecewise_otsu_thresholding(white_keys_vertical_edges)
	# extract bounding boxes with the help of the detected edges using otsu.
	# ASSUMPTION: lines seperating white keys are vertical
	white_keys_bb = extract_white_key_bb(white_key_seperators_mask, black_key_near_distance)
	# generate all keys from the given boundingboxes
	keys_dict, black_keys, white_keys = generate_keys_dict(black_keys_bb, white_keys_bb)

	segmentation = np.zeros_like(image_grayscale)
	# draw segmentation bounding boxes. First draw all white boxes and overlay the black key bounding boxes afterwise, to mimic the disjunctive behaviour
	for white_key_index, white_key_bb in white_keys.items():
		x_min, y_min, x_max, y_max = white_key_bb
		segmentation[y_min:y_max, x_min:x_max] = white_key_index

	for black_key_index, black_key_bb in black_keys.items():
		x_min, y_min, x_max, y_max = black_key_bb
		segmentation[y_min:y_max, x_min:x_max] = black_key_index

	# finding the first fis key in the extracted key makes it possible to offet it accordingly by refferencing the ground truth index of the first fis key
	piano_key_labels = generate_piano_keys()
	first_fis_index = find_first_fis_index(black_keys)
	labels = generate_labels(piano_key_labels, first_fis_index)
	keys_overlay = generate_overlay_segmentation(np.zeros_like(image_bgr), segmentation, labels, black_key_near_distance)
	return keys_overlay, keys_dict, labels, segmentation

def find_hand_and_shadow_mask(background, frame):
	"""finds a segmentation mask

	Args:
		background: background_image (first frame)
		frame: image to detect hands and shadows in 

	Returns:
		overlay: overlayimage displaying the detected hands and shadows
		binary_mask: binary mask masking the hands and shadows
		x_ranges_of_interest: all x ranges hands appear in, to reduce the search space for further processing
	"""
	difference_image = cv2.absdiff(background, frame)

	# Convert difference image to grayscale
	gray_difference = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)

	# Apply thresholding
	_, binary_mask = cv2.threshold(gray_difference, 30, 255, cv2.THRESH_BINARY)
	# Perform morphological operations to remove noise and close holes
	kernel = np.ones((5, 5), np.uint8)
	kernel_stretch_x = np.ones((5, 15), np.uint8)
	kernel_stretch_y = np.ones((15, 5), np.uint8)
	binary_mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
	binary_mask = cv2.morphologyEx(binary_mask_closed, cv2.MORPH_OPEN, kernel_stretch_x)
	binary_mask = cv2.dilate(binary_mask, kernel_stretch_y)
	overlay = cv2.merge([binary_mask, binary_mask, binary_mask])

	_, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
	x = stats[1:, cv2.CC_STAT_LEFT]
	w = stats[1:, cv2.CC_STAT_WIDTH]

	x_ranges_of_interest = ([(x[i], x[i]+w[i]) for i in range(len(x))])
		
	return overlay, binary_mask, x_ranges_of_interest
	

def detect_pressed_keys(keys, x_ranges_of_interest, segmentation, background_image, frame, hand_mask):
	"""detect pressed keys in a given image given its corresponding background image, the keys, their segmentation, the search space and the hand mask

	Args:
		keys: dictionary of index => key bounding box
		x_ranges_of_interest: list of (x_min, x_max) to search in
		segmentation: labeled segmentation mask of keys by index
		background_image: background image (first frame)
		frame: image to detect pressed keys in 
		hand_mask: mask of hands and shadows which occlude or shade the keyboard 

	Returns:
		result: list of key indices which are considered pressed
		metrics_w: metrics for white keys for optimization purposes
		metrics_b: metrics for black keys for optimization purposes
	"""
	result = []
	candidates = []
	metrics_w = []
	metrics_b = []

	# generate a difference image, where positive values correspond to inserted black pixels and negative values to inserted white pixels
	inserted_black_positive = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY).astype(np.int16) - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
	# invert it to apply the oppostie of the behaviour above
	inserted_white_positive = inserted_black_positive * -1
	# clip to only consider the positive values respectivly, such that both difference images correspond to either white pixel changes or black pixel changes
	inserted_black_positive = np.clip(inserted_black_positive, 0, 255).astype(np.uint8)
	inserted_white_positive = np.clip(inserted_white_positive, 0, 255).astype(np.uint8)
	kernel = np.ones((3,3), np.uint8)
	inserted_white_positive = cv2.morphologyEx(inserted_white_positive, cv2.MORPH_OPEN, kernel)
	inserted_black_positive = cv2.morphologyEx(inserted_black_positive, cv2.MORPH_OPEN, kernel)
	
	mask_rgb = ((255 - hand_mask) / 255).astype(np.uint8)

	# inserted black is positive
	inserted_black_positive = inserted_black_positive * mask_rgb
	# inserted white is positive
	inserted_white_positive = inserted_white_positive * mask_rgb

	if args.difference_images:
		cv2.imshow("inserted white filtered", inserted_white_positive)
		cv2.imshow("inserted black filtered", inserted_black_positive)

	if len(x_ranges_of_interest) == 0:
		return candidates, result, metrics_w, metrics_b

	def check_x_overlap(range1, range2):
		return range1[0] < range2[1] and range1[1] > range2[0]
	
	# for each key, check if it overlaps with any range containing hands, if so, add it to the potential candidates of keys, which can be pressed
	for i in range(len(keys)):
		data = keys[i]['data']
		[x1, y1, x2, y2] = data
		for x_range in x_ranges_of_interest:
			if check_x_overlap((x1,x2), x_range):
				candidates.append(i)
				break

	hand_removed_key_segmentation = segmentation * (1 - (hand_mask / 255).astype(np.uint8))
	# kernel is wider in x direction to emphasize vertical line detection
	kernel = np.ones((4, 6), np.uint8)
	for i in candidates:
		# extract segmentation of key
		mask = hand_removed_key_segmentation == i
		# expand key mask to detect neighbouring edges more reliably
		mask_dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel) == 255
		key = keys[i]
		# when the key is white, inserted black pixels are considered, as pressing a white key, makes neighboring black keys as well as the gaps between white keys
		# more prominent and thus the edges darker.
		# If the key is black, we do the opposite, as pressing a black key would imply white pixels appearing on the edges. If the average change within the key
		# change above a certain threshold, consider it pressed
		if key['type'] == 'white':
			metric = np.sum(inserted_black_positive[mask_dilated]) / np.sum(mask)
			metrics_w.append(metric)
			if metric > args.threshold_w:
				result.append(i)
		else:
			metric = np.sum(inserted_white_positive[mask_dilated]) / np.sum(mask)
			metrics_b.append(metric)
			if metric > args.threshold_b:
				result.append(i)
		
	return candidates, result, metrics_w, metrics_b

def generate_mask_for_keys(segmentation, keys):
	return np.isin(segmentation, keys)

def generate_pressed_keys_overlay(segmentation, pressed_keys):
	mask = generate_mask_for_keys(segmentation, pressed_keys) * 255
	return cv2.merge([mask, mask, np.zeros_like(mask)]).astype(np.uint8)	

# Open the video file
cap = cv2.VideoCapture(args.input)

# Check if the video file is opened successfully
if not cap.isOpened():
	print("Error: Could not open video file.")
	exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
	fps = 30

frame_interval = 1.0 / fps

is_background_set = False
background_image = None
overlay = None
keys = None
labels = None
segmentation = None

keys_hist_w = np.array([])
keys_hist_b = np.array([])

while True:
	start_time = time.time()
	# Read the frame
	ret, frame = cap.read()
	# Check if the frame is read successfully
	if not ret:
		break
	
	result = frame.copy()
	if (not is_background_set):
		is_background_set = True
		background_image = frame
		overlay, keys, labels, segmentation = process_background_image(background_image)
	else:
		detected_hands, mask, x_ranges_of_interest = find_hand_and_shadow_mask(background_image, frame)
		candidates, pressed_keys, metrics_w, metrics_b = detect_pressed_keys(keys, x_ranges_of_interest, segmentation, background_image, frame, mask)
		unoccluded_frame_mask = np.logical_not(generate_mask_for_keys(segmentation, candidates))
		pressed_keys_overlay = generate_pressed_keys_overlay(segmentation, pressed_keys)
		background_image = np.where(unoccluded_frame_mask[..., None] != 0, frame, background_image)
		result =  add_without_blend(result, pressed_keys_overlay)
		keys_hist_w = np.concatenate((keys_hist_w, np.array(metrics_w)))
		keys_hist_b = np.concatenate((keys_hist_b, np.array(metrics_b)))
		if args.hand_mask:
			cv2.imshow('Hand and Shadow-mask', mask)

	result = add_without_blend(result,overlay)

	# Display the overlayed frame
	cv2.imshow('Overlayed Video', result)
	cv2.imshow('Original', frame)
	

	# Wait for a key press to exit or continue processing
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	processing_time = time.time() - start_time
	wait_time = max(int((frame_interval - processing_time) * 1000), 1)
	cv2.waitKey(wait_time)

if args.show_hist_metrics:
	fig, axs = plt.subplots(2,1, figsize=(10,12))
	axs[0].hist(keys_hist_w, bins=100, log=True)
	axs[0].set_title('White key brightness difference distribution')
	axs[1].hist(keys_hist_b, bins=100, log=True)
	axs[1].set_title('Black key brightness difference distribution')
	plt.show()

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()