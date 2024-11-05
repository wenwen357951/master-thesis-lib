from scipy import ndimage

def resize_volume(size=(512, 512, 25)):
    def wrap(image):
        """Resize across z-axis"""
        # Set the desired depth
        desired_height = size[0]
        desired_width = size[1]
        desired_depth = size[2]
        # Get current depth
        current_height = image.shape[0]
        current_width = image.shape[1]
        current_depth = image.shape[-1]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # Resize across z-axis
        image = ndimage.zoom(
            image,
            (height_factor, width_factor, depth_factor),
            order=1
        )
        return image
    return wrap
