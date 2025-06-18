# Get altitudes from exif for all files.
for file in *_1.tif; do altitude=$(exiftool -s3 -GPSAltitude "$file" | sed 's/ Above Sea Level//'); echo "$file : $altitude"; done > altitudes.txt

# Copy all tif files from subdirs to the annotated directory.
find . -type f -name '*.tif' -exec cp {} ../../annotated/images/hamburg_mapping_boats/ \;

# Create empty labels for all tif files.
for img in *; do touch "../labels/${img%.tif}.txt"; done