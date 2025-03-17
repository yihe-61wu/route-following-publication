import os
import numpy as np
import cv2

# Classes from the provided code

class SingleOutputMB:
    def __init__(self, N_pn = 1050, N_kc = 20000, N_pn_perkc=10, sparsity_kc=0.1, N_memory=None, output_familiarity=True):
        self._flyhash = isinstance(N_pn_perkc, int)
        self._plastic = N_memory is None
        self._out_fam = output_familiarity

        self.N_pn = N_pn
        self.N_kc = N_pn if not isinstance(N_kc, int) else N_kc

        if self._flyhash:
            self.hash_met = 'FlyHash'
            self.S_kc = sparsity_kc
            self.W_pn2kc = (np.random.rand(self.N_pn, self.N_kc) <= (N_pn_perkc / self.N_pn)).astype(np.int8)
            self.N_kc_WTA = int(self.S_kc * self.N_kc)

        if self._plastic:
            self.W_kc2mbon = np.random.randn(self.N_kc) / np.sqrt(self.N_kc)
            # self.W_kc2mbon = np.zeros(self.N_kc)

    def hashing(self, pn):
        z = np.dot(pn, self.W_pn2kc)
        idx = np.argsort(z)[-self.N_kc_WTA:]
        kc = np.zeros(self.N_kc, dtype=np.int8)
        kc[idx] = 1
        return kc

    def evaluating(self, kc):
        in_mbon = np.dot(kc, self.W_kc2mbon) / self.N_kc
        valence = np.maximum(in_mbon, 0)
        return valence

    def learning(self, kc):
        if self._plastic:
            dW = kc * 2 - 1
            self.W_kc2mbon = self.W_kc2mbon + dW if self._out_fam else self.W_kc2mbon - dW

class LateralisedVisualSensor:
    def __init__(self, view_HW=(30, 50), eye_W=35, grey=True):
        self.view_HW = view_HW
        self.view_size = np.multiply(*view_HW)
        self.grey = grey
        self.eye_W = eye_W
        self.eye_size = self.view_HW[0] * self.eye_W
        self.eye_lb_max = self.view_HW[1] - self.eye_W
        self.mid_eye_lb = self.eye_lb_max // 2
        self.frame = None

    def rgbframe2view(self, rgbframe):
        view = cv2.resize(rgbframe, tuple(np.flip(self.view_HW)))
        if self.grey:
            view = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        return view

    def get_1eye_view(self, offset='l', frame=None):
        if frame is not None:
            self.frame = frame
        view = self.rgbframe2view(self.frame)
        if isinstance(offset, int):
            eye_lb = np.clip(self.mid_eye_lb + offset, 0, self.eye_lb_max)
        elif offset == 'l':
            eye_lb = 0
        elif offset == 'r':
            eye_lb = self.eye_lb_max
        view = view[:, eye_lb : eye_lb + self.eye_W]
        return view

class LaMB:
    def __init__(self,
                 MB_l: SingleOutputMB,
                 MB_r: SingleOutputMB,
                 OL: LateralisedVisualSensor):
        self.OL = OL
        self.MB = {'l': MB_l, 'r': MB_r}

    def _view2familiarity_1MB(self, MB_parity, view_offset, learning):
        MB = self.MB[MB_parity]
        pn = self.OL.get_1eye_view(view_offset).flatten()
        kc = MB.hashing(pn)
        fam = MB.evaluating(kc)
        if learning: MB.learning(kc)
        # change order of evaluating and learning should not affect performance a lot
        return fam

    def view2familiarity(self, view_l_offset, view_r_offset, learning):
        fam_l, fam_r = [self._view2familiarity_1MB(p, o, learning) for p, o in zip('lr', (view_l_offset, view_r_offset))]
        return fam_l, fam_r

def calculate_familiarity(baseline_frame, test_frame):
    # Initializing the components
    MB_l = SingleOutputMB(1050, 20000)
    MB_r = SingleOutputMB(1050, 20000)
    sensor = LateralisedVisualSensor()
    lamb_model = LaMB(MB_l, MB_r, sensor)
    
    sensor.frame = baseline_frame
    _ = lamb_model.view2familiarity('l', 'r', learning=True)

    sensor.frame = test_frame

    fam_l, fam_r = lamb_model.view2familiarity('l', 'r', learning=False)
    
    return fam_l, fam_r

def process_images(directory, baseline_image_path):
    # Read the baseline image
    baseline_image = cv2.imread(baseline_image_path)
    if baseline_image is None:
        raise ValueError(f"Failed to load baseline image from path: {baseline_image_path}")

    # Get list of all files in the directory
    all_files = os.listdir(directory)

    # Filter out only .jpg files
    image_files = [file for file in all_files if file.endswith('.jpg')]

    # Sort the image files to process them in order
    image_files.sort()

    with open('/home/hakosaki/Dissertation_Aug16/MSE_Between_RainyAndSunny_Days/familiarity_data.txt', 'w') as f:
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            test_image = cv2.imread(image_path)
            if test_image is None:
                print(f"Failed to load test image from path: {image_path}")
                continue

            fam_l, fam_r = calculate_familiarity(baseline_image, test_image)
            print(f"For {image_file}:")
            print("Familiarity for left view:", fam_l)
            print("Familiarity for right view:", fam_r)
            print("-----")
            
            # Save to the file
            f.write(f"{fam_l},{fam_r}\n")

# Testing the function
baseline_image_path = "baseline.jpg"
directory_path = "/home/hakosaki/Dissertation_Aug16/MSE_Between_RainyAndSunny_Days/input"  # Replace with your directory path
process_images(directory_path, baseline_image_path)

# # Testing the function
# image_path1 = "Aug14.jpg"
# image_path2 = "Aug14.jpg"
# image1 = cv2.imread(image_path1)
# image2 = cv2.imread(image_path2)
# if image1 is None:
#     raise ValueError(f"Failed to load image from path: {image_path1}")
# if image2 is None:
#     raise ValueError(f"Failed to load image from path: {image_path2}")
# fam_l, fam_r = calculate_familiarity(image1, image2)

# print("Familiarity for left view:", fam_l)
# print("Familiarity for right view:", fam_r)