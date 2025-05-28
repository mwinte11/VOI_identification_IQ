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
import argparse
from typing import List, Tuple, Dict, Any

def read_dicom_folder(folder_path: str) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Read all DICOM files in a folder and return sorted image arrays and metadata."""
    files = sorted(os.listdir(folder_path))
    image_order_array = []
    image_info = []
    rescaled_images = []
    for file in files:
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
    # Sort images by slice location
    sorted_indices = np.argsort(image_order_array)
    rescaled_images_sorted = [rescaled_images[i] for i in sorted_indices]
    image_info_sorted = [image_info[i] for i in sorted_indices]
    return rescaled_images_sorted, image_info_sorted

def circular_mask(array: np.ndarray, centery: int, centerx: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return a circular mask and the pixels within the circle."""
    X, Y = np.ogrid[:array.shape[0], :array.shape[1]]
    dist_from_center = np.sqrt((X - centerx) ** 2 + (Y - centery) ** 2)
    mask = dist_from_center <= radius
    circular_region = np.where(mask, array, np.nan)
    circular_pixels = array[mask]
    return circular_region, circular_pixels

class ClickCapture:
    """
    GUI tool to allow a user to click on an image to select points.
    Returns a list of (x, y, z) tuples.
    """
    def __init__(self, images: List[np.ndarray], start_slice: int = 0):
        self.coordinates = []
        self.images = images
        self.current_slice = start_slice
        self.slice_count = len(images)
        self.global_min = np.min([np.min(image) for image in images])
        self.global_max = np.max([np.max(image) for image in images])
        self.root = tk.Tk()
        self.root.title("Click on the image to select points. Use mouse wheel to scroll slices.")
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
        for i in range(256):
            color = int(i)
            canvas.create_line(0, 255 - i, 50, 255 - i, fill=f'#{color:02x}{color:02x}{color:02x}')
        min_label = tk.Label(legend_frame, text=f"{self.global_min:.2f}")
        min_label.pack(side=tk.BOTTOM)
        return legend_frame
    def update_image(self):
        norm_img = (self.images[self.current_slice] - self.global_min) / (self.global_max - self.global_min) * 255
        self.image = Image.fromarray(norm_img.astype(np.uint8), mode='L')
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
        self.root.quit()
    def get_coordinates(self) -> List[Tuple[int, int, int]]:
        self.root.mainloop()
        return self.coordinates

def get_circular_pixels(image, center_x, center_y, diameter, pixel_spacing, is_cylinder, n_slices, i_slice, slice_thickness):
    """Compute circle dimensions and get pixels in the circle."""
    radius = diameter / 2
    if is_cylinder:
        circle_radius = radius / pixel_spacing
    elif (radius ** 2 > ((n_slices / 2 - i_slice) * slice_thickness) ** 2):
        circle_radius = math.sqrt(radius ** 2 - ((n_slices / 2 - i_slice) * slice_thickness) ** 2) / pixel_spacing
    else:
        circle_radius = 0
    circle_patch = Circle((center_y, center_x), circle_radius, fill=False, edgecolor='red', linewidth=1)
    circular_region, circular_pixels = circular_mask(image, center_y, center_x, circle_radius)
    return circle_patch, pd.DataFrame(circular_region), circular_pixels, circle_radius

def main():
    parser = argparse.ArgumentParser(description="Extract circular regions from DICOM images and save statistics to Excel.")
    parser.add_argument('dicom_folder', help='Path to the folder containing DICOM files')
    parser.add_argument('--output_dir', default='output', help='Directory to save outputs')
    parser.add_argument('--pixel_spacing', type=float, default=1.65, help='Pixel spacing in mm')
    parser.add_argument('--slice_thickness', type=float, default=2.0, help='Slice thickness in mm')
    parser.add_argument('--diameters', nargs='+', type=int, default=[37, 37, 28, 22, 17, 13, 10], help='Diameters of spheres in mm')
    parser.add_argument('--spheres', nargs='+', default=["B0","B1","B2","B3","B4","B5","B6"], help='Names of spheres')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    images, image_info = read_dicom_folder(args.dicom_folder)
    print("Loaded {} image slices.".format(len(images)))

    # User selects center
    click_capture = ClickCapture(images)
    coordinates = click_capture.get_coordinates()
    center_y, center_x, z = coordinates[0]

    nrslices1 = round(args.diameters[1] / args.slice_thickness)
    start_slice1 = int(z - (nrslices1 / 2))

    pd_mean_max = []
    pd_B0 = []
    global_min = np.min([np.min(image) for image in images])
    global_max = np.max([np.max(image) for image in images])

    output_img_dir = os.path.join(args.output_dir, 'BG_output_images')
    Path(output_img_dir).mkdir(exist_ok=True)

    for islice in range(nrslices1+1):
        slice_nr = start_slice1 + islice
        if slice_nr < 0 or slice_nr >= len(images):
            continue
        array_i = images[slice_nr]
        fig, ax = plt.subplots()
        im = ax.imshow(array_i, cmap='gray', vmin=global_min, vmax=global_max)
        circles_added = False
        # Only process for the first sphere (B0)
        diam = args.diameters[1]
        nrslices = round(args.diameters[1] / args.slice_thickness)
        if 0 <= slice_nr < len(images):
            circle, df_circular_region, circular_pixels, array_around_mp = get_circular_pixels(
                array_i, center_x, center_y, diam, args.pixel_spacing, 0, nrslices, islice, args.slice_thickness)
            if array_around_mp != 0:
                ax.add_patch(circle)
                circles_added = True
            pd_B0.extend(circular_pixels.flatten().tolist())
            pd_mean_max.append({
                'Slice NR': slice_nr,
                'Mean': np.mean(circular_pixels),
                'Max': np.max(circular_pixels),
                'Nr Pixels': np.size(circular_pixels),
                'Radius [mm]': array_around_mp * args.pixel_spacing,
                'Sphere': 0
            })
        if circles_added:
            plt.colorbar(im, orientation='vertical', label='Pixel Value [Bq/mL]')
            plt.title(f'Slice NR = {slice_nr}')
            plt.savefig(os.path.join(output_img_dir, f'slice_{slice_nr}.png'))
        plt.close(fig)

    # Save results to Excel
    excel_path = os.path.join(args.output_dir, 'outputMetaBackground.xlsx')
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    pd.DataFrame(pd_mean_max).to_excel(writer, sheet_name='Meta data per slice')
    pd.DataFrame([{
        'Sphere': "B0",
        'Max': np.max(pd_B0),
        'Mean': np.mean(pd_B0),
        'Variance': np.var(pd_B0),
        'Nr Pixels': np.size(pd_B0)
    }]).to_excel(writer, sheet_name='Meta data per sphere')
    writer.save()
    print(f"Results saved to {excel_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_folder", nargs="?", default="path/to/default/folder", help="Path to DICOM folder")
    args = parser.parse_args()
    dicom_folder = args.dicom_folder
    # ...rest of your code...
