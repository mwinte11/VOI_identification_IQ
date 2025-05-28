import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import math
from pathlib import Path
import tkinter as tk
from PIL import Image,ImageTk
from openpyxl import load_workbook

# Function to read DICOM files and extract necessary information
def read_dicom_folder(folder_path):
    files = os.listdir(folder_path)
    files.sort()
    image_order_array = []
    image_info = []
    rescaled_images = []
    nr_slices = 0
    for file in files:
        nr_slices +=1
        image_location = os.path.join(folder_path, file)
        ds = pydicom.dcmread(image_location)
        # print(ds)
        rescale_slope = ds.RescaleSlope
        rescale_intercept = ds.RescaleIntercept
        image_order = ds.SliceLocation
        image_order_array.append(image_order)

        image = ds.pixel_array.astype(np.float64)
        image_rescaled = image * rescale_slope + rescale_intercept
        rescaled_images.append(image_rescaled)

        image_info.append({
            'RescaleSlope': rescale_slope,
            'RescaleIntercept': rescale_intercept,
            'order': image_order
        })

    # Sorting images based on slice location
    sorted_indices = np.argsort(image_order_array)
    rescaled_images_sorted = [rescaled_images[i] for i in sorted_indices]
    image_info_sorted = [image_info[i] for i in sorted_indices]
    # print(nr_slices)
    return rescaled_images_sorted, image_info_sorted

def circular_mask(array_i, centery, centerx, array_around_mp): # create a circular array
    # Create a grid of the same size as the pixel data
    X, Y = np.ogrid[:array_i.shape[0], :array_i.shape[1]]

    # Calculate the distance from the center
    dist_from_center = np.sqrt((X - centerx) ** 2 + (Y - centery) ** 2)

    # Create a mask for the circle
    circular_mask = dist_from_center <= array_around_mp
    circular_region = np.where(circular_mask, array_i, np.nan)
    circular_pixels = array_i[circular_mask]
    return circular_region, circular_pixels

class ClickCapture:
    def __init__(self, images):
        self.coordinates = []
        self.images = images
        self.current_slice = slice_choice
        self.slice_count = len(images)

        # Calculate global min and max pixel values
        self.global_min = np.min([np.min(image) for image in images])
        self.global_max = np.max([np.max(image) for image in images])

        self.root = tk.Tk()
        self.root.title(
            "Start = slice 30. Click on the image to guess the midpoints of background sphere, scroll to see other slices")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        self.label = tk.Label(self.main_frame)
        self.label.pack(side=tk.LEFT)

        self.slice_label = tk.Label(self.root, text=f"Slice: {self.current_slice + 1}/{self.slice_count}")
        self.slice_label.pack(side=tk.BOTTOM)

        self.colorbar_legend = self.create_colorbar_legend()
        self.colorbar_legend.pack(side=tk.RIGHT, fill=tk.Y)

        self.update_image()
        self.label.bind("<Button-1>", self.on_click)
        self.root.bind("<MouseWheel>", self.on_mousewheel)

    def create_colorbar_legend(self):
        legend_frame = tk.Frame(self.main_frame)
        canvas = tk.Canvas(legend_frame, width=50, height=256)
        canvas.pack()

        max_label = tk.Label(legend_frame, text=f"{self.global_max:.2f}")
        max_label.pack(side=tk.TOP)

        # Create gradient for colorbar
        for i in range(256):
            color = int(i)
            canvas.create_line(0, 255 - i, 50, 255 - i, fill=f'#{color:02x}{color:02x}{color:02x}')

        # Add min and max labels
        min_label = tk.Label(legend_frame, text=f"{self.global_min:.2f}")
        min_label.pack(side=tk.BOTTOM)

        return legend_frame

    def update_image(self):
        normalized_image = (self.images[self.current_slice] - self.global_min) / (
                    self.global_max - self.global_min) * 255
        self.image = Image.fromarray(normalized_image.astype(np.uint8), mode='L')
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.label.config(image=self.tk_image)
        self.slice_label.config(text=f"Slice: {self.current_slice + 1}/{self.slice_count}")

    def next_slice(self, event=None):
        if self.current_slice < self.slice_count - 1:
            self.current_slice += 1
            self.update_image()

    def prev_slice(self, event=None):
        if self.current_slice > 0:
            self.current_slice -= 1
            self.update_image()

    def on_mousewheel(self, event):
        if event.delta > 0:
            self.prev_slice()
        else:
            self.next_slice()

    def on_click(self, event):
        z = self.current_slice
        self.coordinates.append((event.x, event.y, z))
        print(f"Clicked at: ({event.x}, {event.y}, {z})")
        if len(self.coordinates) == 1:
            print(f"Clicks recorded: {self.coordinates}")
            self.root.quit()

    def get_coordinates(self):
        self.root.mainloop()
        return self.coordinates

def Circular_Pixels(image, center_x, center_y, diam, pixelspacing, cilinder, nrslices, i, slice_thickness):
    radius = diam / 2  # radius [mm]
    if cilinder == 1:
        array_around_mp = radius / pixelspacing
    elif (radius ** 2 > ((nrslices / 2 - i) * slice_thickness) ** 2):
        array_around_mp = math.sqrt(radius ** 2 - ((nrslices / 2 - i) * slice_thickness) ** 2) / pixelspacing
    else:
        array_around_mp = 0
    circle = Circle((center_y, center_x), array_around_mp, fill=False, edgecolor='red', linewidth=1)
    (circular_region, circular_pixels) = circular_mask(image, center_y, center_x, array_around_mp)
    df_circular_region = pd.DataFrame(circular_region) # Convert the 2D circular region to a DataFrame
    return circle, df_circular_region, circular_pixels, array_around_mp#

# Main function - here is where you have to put in your information
folder_path = './TBR5/PET WB EARL2.0_7mm_ONCO_00003233'
OF = 'OF'
Ampl = 'A07'
TBR = 'T5'
filename = 'PET_'+OF+'_'+Ampl+'_'+TBR+'_'+'targets.xlsx'
filepath = 'PET_'+OF+'_'+Ampl+'_'+TBR+'/'
filenamepath = filepath+filename
if not Path(filepath).exists():
    Path(filepath).mkdir(parents=True)

images, image_info = read_dicom_folder(folder_path)
writer = pd.ExcelWriter(filenamepath, engine='openpyxl')

diams = [37, 37, 28, 22, 17, 13, 10]
spheres = ["B0","B1","B2","B3","B4","B5","B6"]
pixelspacing = 1.65 # mm
slice_thickness = 2 #mm
cilinder = 0 # if cilinder = 1, circles around center slice are all equal size. if cilinder =!1, circles around center slice are the size of radius of circle
nrslices1 = round(diams[1] / slice_thickness) # rounding up nr of slices
slice_choice = 55 # 120#

# Display the center slice image to click on midpoint guesses
# Create an instance of the ClickCapture class
click_capture = ClickCapture(images)
# Start the click capture and get the coordinates
coordinates = click_capture.get_coordinates()
[(center_y1, center_x1, z1)] = coordinates
slice_choice = z1
start_slice1 = int(z1 - (nrslices1 / 2))

pd_mean_max = []
pd_all_0 = []; pd_B0=[]

# Directory to save images
output_dir = filepath+'/BG_output_images'
if not Path(output_dir).exists():
    Path(output_dir).mkdir(parents=True)

# Process each sphere and slice and accumulate circles
global_min = np.min([np.min(image) for image in images])
global_max = np.max([np.max(image) for image in images])

# circular_pixels_0=[]; circular_pixels_1=[]; circular_pixels_2=[]; circular_pixels_3=[]
for islice in range(nrslices1+1):
    print('i =', islice, ', slice NR =', (start_slice1 + islice))
    slice_nr = (start_slice1 + islice) # nr of slice, where islice runs from 0 to nr of slices of sphere 1, slice_nr is the nr of the slice in whole phantom image
    array_i = images[slice_nr]
    fig, ax = plt.subplots()
    im = ax.imshow(array_i, cmap='gray', vmin=global_min,
                   vmax=global_max)  # gray instead of colour & pixel intensities normalized
    circles_added = False
    # plt.imshow(array_i, cmap='gray')
    for isphere in range(1):
        diam = diams[isphere+1]
        center_y, center_x , z = coordinates[isphere]
        nrslices = round(diams[isphere+1] / slice_thickness)
        start_slice = int(z - (nrslices / 2))
        # print("start slice =", start_slice, "center slice =", center_slice, "nr of slices =", nrslices)
        if start_slice <= slice_nr <= slice_choice+(nrslices/2):
            circle, df_circular_region, circular_pixels, array_around_mp = Circular_Pixels(array_i, center_x, center_y, diam, pixelspacing, cilinder, nrslices, islice, slice_thickness) #
        # Add the Circle patch to the plot
            if array_around_mp!=0:
                print(f"Adding circle to slice {slice_nr}...")
                ax.add_patch(circle)
                circles_added = True
            if isphere == 0:  pd_B0.extend(circular_pixels.flatten().tolist())
            pd_mean_max.append(
                {'Slice NR': (slice_nr), 'Mean': np.mean(circular_pixels), 'Max': np.max(circular_pixels), 'Nr Pixels': np.size(circular_pixels),
                 'Radius [mm]': array_around_mp * pixelspacing, 'Sphere': isphere })

    if circles_added:
        plt.colorbar(im, orientation='vertical', label='Pixel Value [Bq/mL]')
        plt.title(f'Slice NR = {slice_nr}')
        plt.savefig(f'{output_dir}/slice_{slice_nr}.png')
        # plt.show()

filenamepath = filepath+'outputMetaBackground.xlsx'
writer = pd.ExcelWriter(filenamepath, engine='openpyxl')

sphere_data =  pd_B0
pd_mean_max_sphere =[]# .append({'Sphere': 'B1', 'Max': np.max(pd_B1), 'Mean': np.mean(pd_B1),
#                            'Variance': np.var(pd_B1), 'Nr Pixels': np.size(pd_B1) })

pd_mean_max_sphere.append({
    'Sphere': "B0",
    'Max': np.max((sphere_data)),
    'Mean': np.mean((sphere_data)),
    'Variance': np.var((sphere_data)),
    'Nr Pixels': np.size((sphere_data))
})

# rewrite metadata to dataframe and also save in excel
df_mean_max = pd.DataFrame(pd_mean_max)
df_mean_max.to_excel(writer, sheet_name='Meta data per slice')
df_mean_max_sphere = pd.DataFrame(pd_mean_max_sphere)
df_mean_max_sphere.to_excel(writer, sheet_name='Meta data per sphere')

writer.save()

print("Circular region pixel data has been saved to ", 'outputMetaBackground.xlsx')