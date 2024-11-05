from matplotlib import pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider
import micasense.imageutils as imageutils
import cv2
import numpy as np

from visualise import get_components_view, get_index_view, get_SR_image, get_PI_image, CHANNEL_NAMES


def show_components_interactive(img_aligned, img_type, img_no="0000"):
    def show_components_view(img, band_indices):

        """Visualise aligned image using the provided band indices."""
        fig, ax = plt.subplots(figsize=(30,24))
        plt.subplots_adjust(bottom=0.25)

        im = ax.imshow(get_components_view(img, band_indices), cmap='gray')

        ax.axis('off')

        return fig, im

    def update_rgb(val):
        """Update the displayed image based on slider values."""
        band_r = int(red_slider.val)
        band_g = int(green_slider.val)
        band_b = int(blue_slider.val)
        band_indices = [band_r, band_g, band_b]

        im.set_data(get_components_view(img_aligned, band_indices))
        fig.canvas.draw_idle()

        # Update slider labels to display the channel names
        red_slider.valtext.set_text(CHANNEL_NAMES[int(red_slider.val)])
        green_slider.valtext.set_text(CHANNEL_NAMES[int(green_slider.val)])
        blue_slider.valtext.set_text(CHANNEL_NAMES[int(blue_slider.val)])

        nonlocal color_type
        color_type = f"{red_slider.valtext.get_text()}_{green_slider.valtext.get_text()}_{blue_slider.valtext.get_text()}"

    def update_index(val):
        """Update the displayed image based on slider values."""
        im.set_data(get_index_view(img_aligned,  int(band1_slider.val), int(band2_slider.val)))
        fig.canvas.draw_idle()

        # Update slider labels to display the channel names
        band1_slider.valtext.set_text(CHANNEL_NAMES[band1_slider.val])
        band2_slider.valtext.set_text(CHANNEL_NAMES[band2_slider.val])

        nonlocal color_type
        color_type = f"index_{band1_slider.valtext.get_text()}_{band2_slider.valtext.get_text()}"
    
    def save_button_callback(event):
        """Callback function for saving the current displayed image."""
        filename = f"out/{img_no}_{color_type}.jpg"

        current_img = im.get_array()
        plt.imsave(filename, current_img, cmap="gray")

        message_text.set_text(f"Saved as {filename}")
        print(f"image saved as {filename}")

        fig.canvas.draw_idle()

    def set_rgb(event):
        red_slider.set_val(2)
        green_slider.set_val(1)
        blue_slider.set_val(0)
        nonlocal color_type 
        color_type = "RGB"

    def set_cir(event):
        red_slider.set_val(3)
        green_slider.set_val(2)
        nonlocal color_type 
        color_type = "CIR"

    def set_ndvi(event):
        band1_slider.set_val(CHANNEL_NAMES.index("NIR"))
        band2_slider.set_val(CHANNEL_NAMES.index("R"))
        nonlocal color_type 
        color_type = "NDVI"

    
    def set_ndwi(event):
        band1_slider.set_val(CHANNEL_NAMES.index("NIR"))
        band2_slider.set_val(CHANNEL_NAMES.index("G"))
        nonlocal color_type 
        color_type = "RNDWI"

    def set_PI(label):
        """Visualise plastic index (PI) image."""
        pi_image = get_PI_image(img_aligned)
        im.set_data(pi_image)
        fig.canvas.draw_idle()
        nonlocal color_type 
        color_type = "PI"

    def set_SR(label):
        """Visualise simple ratio (SR) image."""
        sr_image = get_SR_image(img_aligned)
        im.set_data(sr_image)
        fig.canvas.draw_idle()
        nonlocal color_type
        color_type = "SR"


    fig, im = show_components_view(img_aligned, (2,1,0))
    color_type = "RGB"

    # Create the axis for sliders
    axcolor = 'lightgoldenrodyellow'
    ax_red = plt.axes([0.3, 0.22, 0.25, 0.02], facecolor=axcolor)  # l, b, w,h
    ax_green = plt.axes([0.3, 0.20, 0.25, 0.02], facecolor=axcolor)
    ax_blue = plt.axes([0.3, 0.18, 0.25, 0.02], facecolor=axcolor) 

    # Create sliders for Red, Green, and Blue channel selection
    channels = list(range(img_aligned.shape[2]))
    red_slider = Slider(ax_red, 'Red', 0, len(channels)-2, valinit=2, valstep=1)
    green_slider = Slider(ax_green, 'Green', 0, len(channels)-2, valinit=1, valstep=1)
    blue_slider = Slider(ax_blue, 'Blue', 0, len(channels)-2, valinit=0, valstep=1)

    # Set update function for sliders
    red_slider.on_changed(update_rgb)
    green_slider.on_changed(update_rgb)
    blue_slider.on_changed(update_rgb)

    # Initially set the labels to the current channel names
    red_slider.valtext.set_text(CHANNEL_NAMES[red_slider.val])
    green_slider.valtext.set_text(CHANNEL_NAMES[green_slider.val])
    blue_slider.valtext.set_text(CHANNEL_NAMES[blue_slider.val])

    # SAVE BUTTON
    ax_button = plt.axes([0.60, 0.02, 0.15, 0.04])
    save_button = Button(ax_button, 'Save Image')
    save_button.on_clicked(save_button_callback)
    message_text = plt.text(0.76, 0.03, "", fontsize=12, ha='left', va='center', transform=plt.gcf().transFigure, color="blue")

    # Add buttons for RGB and CIR
    ax_rgb_button = plt.axes([0.60, 0.20, 0.07, 0.04])
    rgb_button = Button(ax_rgb_button, 'RGB')
    rgb_button.on_clicked(set_rgb)

    ax_cir_button = plt.axes([0.68, 0.20, 0.07, 0.04])
    cir_button = Button(ax_cir_button, 'CIR')
    cir_button.on_clicked(set_cir)

    # INDEX sliders
    message_index_formula = plt.text(0.3, 0.15, "Index formula: (band1 - band2) / (band1 + band2)", fontsize=12, ha='left', va='center', transform=plt.gcf().transFigure, color="black")

    ax_band1 = plt.axes([0.3, 0.12, 0.25, 0.02], facecolor=axcolor)  # l, b, w,h
    ax_band2 = plt.axes([0.3, 0.10, 0.25, 0.02], facecolor=axcolor)

    band1_slider = Slider(ax_band1, 'Band 1', 0, len(channels)-2, valinit=3, valstep=1)
    band2_slider = Slider(ax_band2, 'Band 2', 0, len(channels)-2, valinit=2, valstep=1)

    band1_slider.valtext.set_text(CHANNEL_NAMES[band1_slider.val])
    band2_slider.valtext.set_text(CHANNEL_NAMES[band2_slider.val])

    band1_slider.on_changed(update_index)
    band2_slider.on_changed(update_index)

    # Add buttons for NDVI and NDWI
    ax_ndvi_button = plt.axes([0.6, 0.1, 0.07, 0.04])
    ndvi_button = Button(ax_ndvi_button, 'NDVI')
    ndvi_button.on_clicked(set_ndvi)

    ax_ndwi_button = plt.axes([0.68, 0.1, 0.07, 0.04])
    ndwi_button = Button(ax_ndwi_button, 'RNDWI')
    ndwi_button.on_clicked(set_ndwi)

    pi_button = Button(plt.axes([0.3, 0.05, 0.07, 0.04]), 'PI NIR/(NIR+R)')
    pi_button.on_clicked(set_PI)
    
    sr = Button(plt.axes([0.38, 0.05, 0.07, 0.04]), 'SR NIR/R')
    sr.on_clicked(set_SR)

    plt.show()


