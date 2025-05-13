#! /bin/bash
# python3 -m detection.detect_on_all -b "mandrac/bag_files/matriceBag_multispectral_2024-12-06-10-33-18.bag" -o "transparent_sea.mp4" -i "mandrac_2024_12_6" -s 24 -e 339
# python3 -m detection.detect_on_all -b "mandrac/bag_files/matriceBag_multispectral_2024-12-06-11-09-36.bag" -o "green_sea.mp4" -i "mandrac_2024_12_6" -s 30 -e 274
# python3 -m detection.detect_on_all -b "mandrac/bag_files/matriceBag_multispectral_2024-12-06-11-35-53.bag" -o "green_marina.mp4" -i "mandrac_2024_12_6" -s 2
# python3 -m detection.detect_on_all -b "mandrac/bag_files/matriceBag_multispectral_2024-12-06-11-56-00.bag" -o "transparent_marina.mp4" -i "mandrac_2024_12_6" -s 12 -e 113

# python3 -m detection.detect_on_all -b "mandrac2/bags/matriceBag_multispectral_2025-04-04-10-47-38.bag" -o "beach.mp4" -i "mandrac_2025_04_04" -s 79 -e 349
# python3 -m detection.detect_on_all -b "mandrac2/bags/matriceBag_multispectral_2025-04-04-10-54-23.bag" -o "beach_throwing.mp4" -i "mandrac_2025_04_04" -s 684 -e 786 
# python3 -m detection.detect_on_all -b "mandrac2/bags/matriceBag_multispectral_2025-04-04-12-38-42.bag" -o "marina2.mp4" -i "mandrac_2025_04_04" -p "0010" -s 57 -e 397

# python3 -m detection.detect_on_all -b "mandrac3/bags/matriceBag_multispectral_2016-02-11-18-02-40.bag" -o "weights.mp4" -i "mandrac_2025_04_16" -s 21 -e 229
python3 -m detection.detect_on_all -b "mandrac3/bags/matriceBag_multispectral_2016-02-11-18-47-02.bag" -o "pile.mp4" -i "mandrac_2025_04_16" -p "0436" -s 260 -e 433
# python3 -m detection.detect_on_all -b "mandrac3/bags/matriceBag_multispectral_2016-02-11-18-59-05.bag" -o "transparent3.mp4" -i "mandrac_2025_04_16" -p "0436" -s 449 -e 600
python3 -m detection.detect_on_all -b "mandrac3/bags/matriceBag_multispectral_2016-02-11-19-09-56.bag" -o "green3.mp4" -i "mandrac_2025_04_16" -p "0436" -s 602 -e 825
# python3 -m detection.detect_on_all -b "mandrac3/bags/matriceBag_multispectral_2016-02-11-19-25-55.bag" -o "green_and_transparent.mp4" -i "mandrac_2025_04_16" -p "0436" -s 839 -e 1095