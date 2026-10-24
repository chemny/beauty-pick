"""Shared decoder registration for original images in API and worker processes."""
from pillow_heif import register_heif_opener

register_heif_opener()
FORMAT_MIME = {
    'JPEG':'image/jpeg', 'PNG':'image/png', 'WEBP':'image/webp',
    'HEIF':'image/heif', 'AVIF':'image/avif', 'BMP':'image/bmp',
    'TIFF':'image/tiff', 'GIF':'image/gif', 'JPEG2000':'image/jp2',
}
SUPPORTED_IMAGE_FORMATS = ['JPEG', 'PNG', 'WebP', 'HEIC/HEIF', 'AVIF', 'BMP', 'TIFF', '静态 GIF', 'JPEG 2000']
