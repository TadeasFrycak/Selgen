from multiprocessing import cpu_count, Pool
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import selgen_global
import pandas as pd
import numpy as np
import cv2
import sys
import os


def crop_roi(img: np.ndarray, threshold: int = 150):
    """
    Localizes region of interest = black tray with crop based on gradient
    between white background and black borders of tray
    :param img:
    :param threshold:
    """

    assert type(img) == np.ndarray and len(img.shape) == 3 and img.dtype == "uint8", "Input data has wrong data type"

    b, g, r = cv2.split(img)

    # columns
    col_mask = b < threshold
    col_mask[:, col_mask.shape[1] // 2 - 500:col_mask.shape[1] // 2 + 500] = 1

    proj = np.sum(col_mask, axis=0)
    mez = max(proj) / 3

    pos = round(len(proj) / 2)

    while proj[pos] > mez:
        pos = pos + 1

    dos = pos

    pos = round(len(proj) / 2)

    while proj[pos] > mez:
        pos = pos - 1

    ods = pos

    # rows
    row_mask = b < threshold

    proj = np.sum(row_mask, axis=1)
    mez = max(proj) / 3
    pos = round(len(proj) / 2)

    while proj[pos] > mez:
        pos = pos + 1

    dor = pos

    pos = round(len(proj) / 2)

    while proj[pos] > mez:
        pos = pos - 1

    odr = pos

    ROI = img[odr:dor, ods:dos, :]

    return ROI


def find_opt_rotation(mask: np.ndarray):
    (h, w) = mask.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    crit = []
    angle = []

    for a in np.arange(-20, 21):
        M = cv2.getRotationMatrix2D((cX, cY), a, 1.0)
        rotated = cv2.warpAffine(mask, M, (w, h))

        crit.append(np.var(np.sum(rotated, axis=1)) + np.var(np.sum(rotated, axis=1)))
        angle.append(a)

    return angle[np.argmax(crit)]


def find_splits(signal, COUNT):
    # TODO simplify
    def square_function(xo, frequency, duty, offset):
        xo = xo + offset
        period = len(signal) / frequency
        return (xo % period > period * duty) * np.max(signal) * 0.6

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    signal = signal - moving_average(signal, len(signal) // 10)
    signal -= signal.mean()

    x = np.linspace(0, len(signal), len(signal))
    offsets = []
    for offset in np.linspace(-len(x) // COUNT // 2, len(x) // COUNT // 2 + 1, 100):
        for frequency in np.linspace(COUNT, COUNT + 1, 10):
            for duty in [0.8]:
                y = square_function(x, frequency, duty, offset)
                error = np.square(np.subtract(y, signal)).mean()
                offsets.append((frequency, duty, offset, error))

    frequency, duty, offset, error = min(offsets, key=lambda z: z[-1])

    diff = np.diff(square_function(x, frequency, duty, offset))
    diff_tc = np.abs(diff)

    peaks, _ = find_peaks(diff)
    peaks_tc, _ = find_peaks(diff_tc)

    peaks_n = peaks + np.min(np.unique(np.diff(peaks_tc))) // 2

    for n, peak in enumerate(peaks_n):
        if peak >= len(signal) - 1:
            peaks_n[n] = len(signal) - 1

    # freq = np.fft.fftfreq(len(x), (x[1] - x[0]))  # assume uniform spacing
    # Fy = abs(np.fft.fft(y))
    #
    # guess_freq = abs(freq[np.argmax(Fy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    # guess_amp = np.std(y) * 2. ** 0.5
    # guess_offset = np.mean(y)
    # guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])
    #
    # def sinfunc(t, A, w, p, c):
    #     return A * np.sin(w * t + p) + c
    #
    # popt, pcov = curve_fit(sinfunc, x, y, p0=guess, maxfev=10000)
    # A, w, p, c = popt
    # fitfunc = lambda t: A * np.sin(w * t + p) + c
    #
    # return fitfunc(x)

    return peaks_n


def get_grid_coords(img: np.ndarray):
    """
    Identifies coordinates of grid on given part of tray with grid mask
    """

    assert type(img) == np.ndarray and len(img.shape) == 3 and img.dtype == "uint8", "Input data has wrong data type"

    h, w = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    opt_angle = find_opt_rotation(img)

    M = cv2.getRotationMatrix2D((cX, cY), opt_angle, 1.0)
    img = cv2.cvtColor(cv2.warpAffine(img, M, (w, h)), cv2.COLOR_BGR2GRAY)

    # Compute signal for rows and columns
    row_signal = np.sum(img, axis=1)
    col_signal = np.sum(img, axis=0)

    row_peaks = find_splits(row_signal, 7)
    col_peaks = find_splits(col_signal, 9)

    # plt.plot(row_signal)
    # plt.plot(row_peaks, row_signal[row_peaks], "x")
    # plt.show()

    # plt.plot(col_signal)
    # plt.plot(col_peaks, col_signal[col_peaks], "x")
    # plt.show()

    return row_peaks, col_peaks


def split_cells(img: np.ndarray, row_indexes: np.ndarray, col_indexes: np.ndarray):
    """
    Separates part of tray with identified coordinates in cells of crop growth
    """
    assert type(img) == np.ndarray and len(img.shape) == 3 and img.dtype == "uint8", "Input data has wrong data type"

    areas = []

    class Area:
        def __init__(self, row, column, cropped_area, size):
            self.row = row
            self.column = column
            self.cropped_area = cropped_area
            self.size = size

    for i in range(0, len(row_indexes) - 1):
        for j in range(0, len(col_indexes) - 1):
            cropped_area = img[row_indexes[i]:row_indexes[i + 1], col_indexes[j]:col_indexes[j + 1], :]
            area_ = Area(i, j, cropped_area, cropped_area.shape[0:2])
            areas.append(area_)

    return areas


def process_image(img):
    """
    Performs all previous functions/steps and returns tray area and list of crop growth cells for left and right
    part this function also paints grid of tray with red lines for image visual result
    """

    assert type(img) == np.ndarray and len(img.shape) == 3 and img.dtype == "uint8", "Input data has wrong data type"

    roi = crop_roi(img)
    rows, cols = get_grid_coords(roi)

    if len(rows) != 7 or len(cols) != 10:
        raise Exception('Grid structure of tray wasn\'t found')

    areas = split_cells(roi, rows, cols)

    for line in rows:
        cv2.line(roi, (cols[0], line), (cols[-1], line), (255, 0, 0), 2)

    for line in cols:
        cv2.line(roi, (line, rows[0]), (line, rows[-1]), (255, 0, 0), 2)

    return areas, roi


def segmentation_biomass(img, lower_thresh, upper_thresh):
    """
    Segments crop in given area with predefined thresholds in HSV color space and returns biomass
    """
    assert type(img) == np.ndarray and len(img.shape) == 3 and img.dtype == "uint8", "Input data has wrong data type"

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = (mask > 0).astype('uint8')

    return mask.sum() / (h * w)


def paint_active_biomass(img, lower_thresh, upper_thresh):
    """
    Draws contours around active crop biomass in a whole image
    """
    assert type(img) == np.ndarray and len(img.shape) == 3 and img.dtype == "uint8", "Input data has wrong data type"

    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = (mask > 0).astype('uint8') * 255
    mask = cv2.Canny(mask, 100, 200)

    cnts, __ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, cnts, -1, (0, 0, 255), 1)

    return img


def filename_parser(file: str) -> dict:
    """
    Parses important data from image filename
    """
    info = file.split('.')[0].split('_')
    return {'variant': info[0], 'date': info[2], 'time': info[4]}


def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]


def process_images(img):
    print("[INFO] Starting process {}".format(img["id"]))
    print("[INFO] For process {}, there is {} images in processing queue".format(img["id"],
                                                                                 len(img["files_names"])))

    f = open(img["temp_path"] + "failures_{}.txt".format(img["id"]), "w+")
    final_data = []

    for imageName in img["files_names"]:
        try:

            metadata = filename_parser(imageName)

            image = cv2.imread(img["input_path"] + imageName)

            areas, roi = process_image(image)

            contoured_roi = paint_active_biomass(roi, selgen_global.lower_thresh, selgen_global.upper_thresh)

            cv2.imwrite(img["output_path"] + imageName, contoured_roi)

            image_data = []

            for area in areas:
                biomass = segmentation_biomass(area.cropped_area, selgen_global.lower_thresh,
                                               selgen_global.upper_thresh)

                image_data.append(
                    dict(zip(('filename', 'date', 'time', 'variant', 'row', 'column', 'biomass', 'size'),
                             (imageName, metadata['date'], metadata['time'], metadata['variant'], area.row,
                              area.column, biomass, area.size))))

            final_data = final_data + image_data

        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print('{} - {}: {}'.format(filename, line_number, exception_object))
            f.write('{} - {}: {}\n'.format(filename, line_number, exception_object))

        df = pd.DataFrame(final_data)
        df = df[['filename', 'date', 'time', 'variant', 'row', 'column', 'biomass', 'size']]
        df.sort_values(by=['row', 'column', 'date', 'time'])
        df.to_excel(img["temp_path"] + '/batch_result_{}.xlsx'.format(img["id"]))


if __name__ == '__main__':

    assert os.path.exists(selgen_global.path), 'path should navigate into the folder where batch of images are stored'

    temp_path = selgen_global.path + 'temp/'
    output_path = selgen_global.path + 'results/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    FORMATS = ('.JPG', '.jpg', '.PNG', '.png', '.bmp', '.BMP', '.TIFF', '.tiff', '.TIF', '.tif')
    files = [file for file in os.listdir(selgen_global.path) if file.endswith(FORMATS)]
    data = []

    procs = cpu_count()
    procIDs = list(range(0, procs))

    numImagesPerProc = len(files) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))

    chunkedPaths = list(chunk(files, numImagesPerProc))

    # initialize the list of payloads
    imageLoads = []

    # loop over the set chunked image paths
    for (i, fileNames) in enumerate(chunkedPaths):
        # construct a dictionary of data for the payload, then add it
        # to the payloads list
        data = {
            "id": i,
            "files_names": fileNames,
            "input_path": selgen_global.path,
            "output_path": output_path,
            "temp_path": temp_path
        }
        imageLoads.append(data)

    structured_data = []

    # construct and launch the processing pool
    print("[INFO] Launching pool using {} processes.".format(procs))
    print(
        "[INFO] All CPU capacity is used for data analysis. You won't be able to use your computer for any other actions.")

    pool = Pool(processes=procs)
    pool.map(process_images, imageLoads)
    pool.close()
    pool.join()

    print("[INFO] Pool of processes was closed")
    print("[INFO] Aggregating partial results into structured data set.")

    xlsx_files = [file for file in os.listdir(temp_path) if file.endswith('xlsx')]
    txt_files = [file for file in os.listdir(temp_path) if file.endswith('txt')]

    frames = []

    for xlsx in xlsx_files:
        frames.append(pd.read_excel(temp_path + xlsx, engine='openpyxl'))

    structured_result = pd.concat(frames, ignore_index=True)
    structured_result = structured_result[['filename', 'date', 'time', 'variant', 'row', 'column', 'biomass', 'size']]
    structured_result.sort_values(by=['row', 'column', 'date', 'time'])
    structured_result.to_excel(output_path + 'exp_result.xlsx', index=False)

    with open(output_path + 'failures.txt', 'w') as outfile:
        for fname in txt_files:
            with open(temp_path + fname) as infile:
                for line in infile:
                    outfile.write(line)

    files = [file for file in os.listdir(temp_path)]

    for f in files:
        os.remove(temp_path + f)

    os.rmdir(temp_path)

    print("[INFO] ANALYSIS WAS FINISHED")
