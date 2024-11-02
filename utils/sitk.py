import SimpleITK as sitk


def sitk_show(_image_arr, _name):
    _image = sitk.GetImageFromArray(_image_arr)
    sitk.Show(_image, title=_name)
