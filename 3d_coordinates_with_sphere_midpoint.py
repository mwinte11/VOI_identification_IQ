#%%
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import math
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

# Function to read DICOM files and extract necessary information
def read_dicom_folder(folder_path):
    files = os.listdir(folder_path)
    files.sort()
    image_order_array = []
    image_info = []
    rescaled_images = []
    nr_slices = 0
    for file in files:
        nr_slices += 1
        image_location = os.path.join(folder_path, file)
        ds = pydicom.dcmread(image_location)
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
    sorted_indices = np.argsort(image_order_array)
    rescaled_images_sorted = [rescaled_images[i] for i in sorted_indices]
    image_info_sorted = [image_info[i] for i in sorted_indices]
    return rescaled_images_sorted, image_info_sorted

# Function to find the slice with highest mean intensity; center_slice
def find_center_slice(images):
    mean_values = [np.mean(image) for image in images]
    max_mean_index = np.argmax(mean_values)
    return max_mean_index, images[max_mean_index]

def circular_mask(array_i, centery, centerx, array_around_mp):
    X, Y = np.ogrid[:array_i.shape[0], :array_i.shape[1]]
    dist_from_center = np.sqrt((X - centerx) ** 2 + (Y - centery) ** 2)
    circular_mask = dist_from_center <= array_around_mp
    circular_region = np.where(circular_mask, array_i, np.nan)
    circular_pixels = array_i[circular_mask]
    return circular_region, circular_pixels

# Function to find the midpoint in x,y,z
def find_midpoint_3d(images, center_x, center_y, center_z, diam, pixelspacing, z_range, list_mean_intensity_sphere=None):
    list_midpoint = []
    list_mean_intensity = []
    ran = 5  # range of pixels around midpoint in x and y direction
    pd_MP_mean_max = []  # pd.DataFrame(columns=['MidPoint', 'Mean', 'Median', 'Max'])
    nrslices = round(diam / 2)  # slice thickness = 2 mm
    
    for zi in range(center_z - z_range, center_z + z_range):
        center_xi = center_x - ran
        start_slice = int(zi - (nrslices / 2))
        end_slice = int(zi + (nrslices / 2))
        print('Start slice is: ', start_slice, ', end slice is: ', end_slice )
        for xi in range(2 * ran):
            center_yi = center_y - ran
            center_xi = center_xi + 1
            for yi in range(2 * ran):
                center_yi = center_yi + 1
                list_midpoint.append([center_xi, center_yi, zi])
                # image = images[zi]
                list_intensity_sphere = []
                for slice_nr in range((zi-nrslices), (zi+nrslices)):
                    if start_slice <= slice_nr <= end_slice:
                        circle, df_circular_region, circular_pixels, array_around_mp = Circular_Pixels(images[slice_nr], center_xi,
                                                                                                       center_yi,
                                                                                                       diam, 1.65,
                                                                                                       0, nrslices,
                                                                                                       slice_nr - start_slice,
                                                                                                       slice_thickness)
                        # print('Slice NR: ', slice_nr, ', mean pixel intensity: ', np.mean(circular_pixels), ', length of circular_pixels:', len(circular_pixels), ', list_mean_intensity:', list_mean_intensity
                        #       , ', list_intensity_sphere:', list_intensity_sphere)
                        # circular_region, circular_pixels = circular_mask(images[slice_nr], center_yi, center_xi, diam / (2 * pixelspacing))
                        list_intensity_sphere.extend(circular_pixels)
                list_mean_intensity.append(np.mean(list_intensity_sphere))
                pd_MP_mean_max.append({'MidPoint x': center_xi, 'MidPoint y': center_yi, 'MidPoint z': zi, 'Mean': np.mean(list_intensity_sphere),
                                       'Max': np.max(list_intensity_sphere)})

    midpoint_index = list_mean_intensity.index(max(list_mean_intensity))
    print(max(list_mean_intensity), list_midpoint[midpoint_index])
    [center_x, center_y, center_z] = list_midpoint[midpoint_index]
    df_MP_mean_max = pd.DataFrame(pd_MP_mean_max)

    return center_x, center_y, center_z, df_MP_mean_max

def Circular_Pixels(image, center_x, center_y, diam, pixelspacing, cilinder, nrslices, i, slice_thickness):
    radius = diam / 2  # radius [mm]
    if cilinder == 1:
        array_around_mp = radius / pixelspacing
    elif (radius**2 > ((nrslices / 2 - i) * slice_thickness)**2):
        array_around_mp = math.sqrt(radius ** 2 - ((nrslices / 2 - i) * slice_thickness) ** 2) / pixelspacing
    else:
        array_around_mp = 0

    circle = Circle((center_y, center_x), array_around_mp, fill=False, edgecolor='red', linewidth=1)
    (circular_region, circular_pixels) = circular_mask(image, center_y, center_x, array_around_mp)
    df_circular_region = pd.DataFrame(circular_region) # Convert the 2D circular region to a DataFrame
    return circle, df_circular_region, circular_pixels, array_around_mp

class ClickCapture:
    def __init__(self, images):
        self.coordinates = []
        self.images = images
        self.current_slice = center_slice
        self.slice_count = len(images)

        # Calculate global min and max pixel values
        self.global_min = np.min([np.min(image) for image in images])
        self.global_max = np.max([np.max(image) for image in images])

        self.root = tk.Tk()
        self.root.title("Start = center slice. Click on the image to guess the midpoints of target spheres 1, 2, 3 and 4, scroll to see other slices")

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
        normalized_image = (self.images[self.current_slice] - self.global_min) / (self.global_max - self.global_min) * 255
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
        if len(self.coordinates) == 6:
            print(f"Six clicks recorded: {self.coordinates}")
            self.root.quit()

    def get_coordinates(self):
        self.root.mainloop()
        return self.coordinates

# Main function - here is where you have to put in your information
folder_path = './TBR5/PET WB EARL2.0_7mm_00003146'
OF = 'NOF'
Ampl = 'A07'
TBR = 'T5'
filename = 'PET_' + OF + '_' + Ampl + '_' + TBR + '_' + 'targets.xlsx'
filepath = 'PET_' + OF + '_' + Ampl + '_' + TBR + '/'
filenamepath = filepath + filename
if not Path(filepath).exists():
    Path(filepath).mkdir(parents=True)

images, image_info = read_dicom_folder(folder_path)
center_slice, center_slice_image = find_center_slice(images)

# Display the center slice image to click on midpoint guesses
click_capture = ClickCapture(images)
coordinates = click_capture.get_coordinates()

writer = pd.ExcelWriter(filenamepath, engine='openpyxl')
(center_y1, center_x1, z1), (center_y2, center_x2, z2), (center_y3, center_x3, z3), (center_y4, center_x4, z4), (center_y5, center_x5, z5), (center_y6, center_x6, z6)  = coordinates
diams = [37, 37, 28, 22, 17, 13, 10]
spheres = ["B0","B1","B2","B3","B4","B5","B6"]
pixelspacing = 1.65 # mm
slice_thickness = 2 #mm
cilinder = 0 # if cilinder = 1, circles around center slice are all equal size. if cilinder =!1, circles around center slice are the size of radius of circle
print('center slice is ',center_slice)
nrslices1 = round(diams[1] / slice_thickness)+2  # rounding up nr of slices
slice_choice = center_slice
start_slice1 = int(slice_choice - (nrslices1 / 2))
print('nrslices=', nrslices1,', startslice=',start_slice1)

pd_mean_max = [] #pd.DataFrame(columns=['Slice NR', 'Mean', 'Max','Radius [mm]','Sphere'])
pd_all_0 = []; pd_B0=[]; pd_B1 = []; pd_B2 = []; pd_B3 = []; pd_B4 = []; pd_B5 = []; pd_B6 =[]
pd_B = [pd_B1, pd_B2, pd_B3, pd_B4, pd_B5, pd_B6]

# Directory to save images
output_dir = filepath+'/output_images'
if not Path(output_dir).exists():
    Path(output_dir).mkdir(parents=True)

background_slice = 54  #center slice 73 background slice 54
sphere_data = pd_B1, pd_B2, pd_B3, pd_B4, pd_B5, pd_B6 #, pd_B0,
pd_mean_max_sphere =[]

# Optimizing midpoints in x,y
filenamepath = filepath + 'outputMidPoints3D.xlsx'
writer2 = pd.ExcelWriter(filenamepath, engine='openpyxl') # save the midpoints and their mean and max pixel intensities per sphere
# ALS TBR == 2: GEBRUIK DAN VASTVOORSCHRIJVEN == 2 of 3
VV = 0

for isphere in range(6):
    diam = diams[isphere + 1]
    center_y, center_x, z_slice = coordinates[isphere]
    if isphere <=(5-VV):
        center_x, center_y, center_z, df_MP_mean_max = find_midpoint_3d(images, center_x, center_y, z_slice, diam, pixelspacing, 2)
        df_MP_mean_max.to_excel(writer2, sheet_name='B0' + str(isphere + 1))
        writer2.save()
    # elif isphere == 4:
    #     center_z = coordinates[3][2]
    #     df_MP_mean_max = 0
    #     center_y = coordinates[2][0]-1
    #     center_x = 175
    # elif isphere == 5:
    #     center_z = coordinates[0][2]
    #     df_MP_mean_max = 0
    #     center_y = coordinates[1][0]-1
    #     center_x = 175
    coordinates[isphere] = center_y, center_x, center_z


# Process each sphere and slice and accumulate circles
global_min = np.min([np.min(image) for image in images])
global_max = np.max([np.max(image) for image in images])
for slice_nr in range((center_slice-10), (center_slice+11)):
    print(f"Processing slice {slice_nr}...")
    array_i = images[slice_nr]
    fig, ax = plt.subplots()
    im = ax.imshow(array_i, cmap='gray', vmin=global_min, vmax=global_max) # gray instead of colour & pixel intensities normalized
    circles_added = False

    for isphere in range(6):
        diam = diams[isphere + 1]
        center_y, center_x, z_slice = coordinates[isphere]
        nrslices = round(diams[isphere + 1] / slice_thickness)
        start_slice = int(z_slice - (nrslices / 2))
        end_slice = int(z_slice + (nrslices / 2))
        print('isphere=', isphere,', nrslices=', nrslices, ', startslice=', start_slice,', endslice=', end_slice, ', slicenr=', slice_nr)

        if start_slice <= slice_nr <= end_slice:
            circle, df_circular_region, circular_pixels, array_around_mp = Circular_Pixels(array_i, center_x, center_y,
                                                                                           diam, pixelspacing, cilinder,
                                                                                           nrslices,
                                                                                           slice_nr - start_slice,
                                                                                           slice_thickness)
            if array_around_mp!=0: # if there are more than 1 pixel on this slice for this sphere
                print(f"Adding circle for sphere {isphere+1} to slice {slice_nr}...")
                ax.add_patch(circle)
                circles_added = True
                # circular_pixels_old=circular_pixels

            pd_B[isphere].extend(circular_pixels.flatten().tolist())
            pd_mean_max.append(
                {'Slice NR': slice_nr, 'Mean': np.mean(circular_pixels), 'Max': np.max(circular_pixels),
                 'Nr Pixels': np.size(circular_pixels),
                 'Radius [mm]': array_around_mp * pixelspacing, 'Sphere': isphere})
            df_circular_region = df_circular_region.dropna(how='all')
            df_circular_region = df_circular_region.dropna(how='all', axis=1)
            sheetname = 'Slice' + str(slice_nr) + 'B0' + str(isphere + 1)
            df_circular_region.to_excel(writer, sheet_name=str(sheetname))  # add excelwriter data
            writer.save()

    if circles_added:
        plt.colorbar(im, orientation='vertical', label='Pixel Value [Bq/mL]')
        plt.title(f'Slice NR = {slice_nr}')
        plt.savefig(f'{output_dir}/slice_{slice_nr}.png')
        # plt.show()
filenamepath = filepath+'outputMetaTargets.xlsx'
writer = pd.ExcelWriter(filenamepath, engine='openpyxl')

# Collect statistics and save each DataFrame to an Excel sheet
for i, sphere_data in enumerate(pd_B):
    sphere_name = f'B{i+1}'
    pd_mean_max_sphere.append({
        'Sphere': sphere_name,
        'Max': np.max(sphere_data),
        'Mean': np.mean(sphere_data),
        'Variance': np.var(sphere_data),
        'Nr Pixels': np.size(sphere_data)
    })

    # Save each DataFrame to an Excel sheet
    df_all = pd.DataFrame(sphere_data, columns=[f'Pixel Value B{i+1}'])
    df_all.to_excel(writer, sheet_name=f'All Pixels B{i+1}')

# Rewrite metadata to DataFrame and save in Excel
df_mean_max = pd.DataFrame(pd_mean_max)
df_mean_max.to_excel(writer, sheet_name='Meta data per slice')

df_mean_max_sphere = pd.DataFrame(pd_mean_max_sphere)
df_mean_max_sphere.to_excel(writer, sheet_name='Meta data per sphere')

writer.save()

print("Circular region pixel data has been saved to ", filenamepath)

# Now 'coordinates' contains (x, y, z) tuples for each click
print(f"Final coordinates: {coordinates}")
