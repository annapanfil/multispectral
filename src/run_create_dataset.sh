#!/bin/bash

# train and test on whole
python -m detection.create_dataset -n "rndwi_whole_random" -f "N # G"
python -m detection.create_dataset -n "meanre_whole_random" -f "(4#1) + (4#0)"
python -m detection.create_dataset -n "RGB_whole_random"

python -m detection.create_dataset -n "ghost-net-1_whole_random" -f "(4 # (4 + 1))"
python -m detection.create_dataset -n "ghost-net-2_whole_random" -f "(4 # ((3 + 2) + 1))"
python -m detection.create_dataset -n "ghost-net-3_whole_random" -f "(2 # 1)"
python -m detection.create_dataset -n "ghost-net-4_whole_random" -f "((1 # 4) * (3 # (((1 / 4) + (2 # (1 / (2 * 0)))) + 1)))"
python -m detection.create_dataset -n "ghost-net-5_whole_random" -f "(((3 # ((((1 + 3) - (2 / 3)) + (0 / 4)) / 3)) * 1) / 1)"
python -m detection.create_dataset -n "ghost-net-6_whole_random" -f "((3 # 1) - (1 # (3 * 2)))"
python -m detection.create_dataset -n "ghost-net-7_whole_random" -f "(4 # (2 + 2))"

python -m detection.create_dataset -n "sea-1_whole_random" -f "((2 + 0) - 4)"

python -m detection.create_dataset -n "pool-1_whole_random" -f "4 # 1"
python -m detection.create_dataset -n "pool-2_whole_random" -f "(4 # ((((2 + 3) + (2 + (2 + 3))) / 0) + 1))"
python -m detection.create_dataset -n "pool-3_whole_random" -f "2 # 1"
python -m detection.create_dataset -n "pool-4_whole_random" -f "((3 * 3) # ((3 + 3) + 1))"
python -m detection.create_dataset -n "pool-5_whole_random" -f "(1 + (2 - 0))"

# train and test on pool
python -m detection.create_dataset -e mandrac -n "rndwi_pool_random" -f "N # G"
python -m detection.create_dataset -e mandrac -n "meanre_pool_random" -f "(4#1) + (4#0)"
python -m detection.create_dataset -e mandrac -n "RGB_pool_random"

python -m detection.create_dataset -e mandrac -n "ghost-net-1_pool_random" -f "(4 # (4 + 1))"
python -m detection.create_dataset -e mandrac -n "ghost-net-2_pool_random" -f "(4 # ((3 + 2) + 1))"
python -m detection.create_dataset -e mandrac -n "ghost-net-3_pool_random" -f "(2 # 1)"
python -m detection.create_dataset -e mandrac -n "ghost-net-4_pool_random" -f "((1 # 4) * (3 # (((1 / 4) + (2 # (1 / (2 * 0)))) + 1)))"
python -m detection.create_dataset -e mandrac -n "ghost-net-5_pool_random" -f "(((3 # ((((1 + 3) - (2 / 3)) + (0 / 4)) / 3)) * 1) / 1)"
python -m detection.create_dataset -e mandrac -n "ghost-net-6_pool_random" -f "((3 # 1) - (1 # (3 * 2)))"
python -m detection.create_dataset -e mandrac -n "ghost-net-7_pool_random" -f "(4 # (2 + 2))"

python -m detection.create_dataset -e mandrac -n "sea-1_pool_random" -f "((2 + 0) - 4)"

python -m detection.create_dataset -e mandrac -n "pool-1_pool_random" -f "4 # 1"
python -m detection.create_dataset -e mandrac -n "pool-2_pool_random" -f "(4 # ((((2 + 3) + (2 + (2 + 3))) / 0) + 1))"
python -m detection.create_dataset -e mandrac -n "pool-3_pool_random" -f "2 # 1"
python -m detection.create_dataset -e mandrac -n "pool-4_pool_random" -f "((3 * 3) # ((3 + 3) + 1))"
python -m detection.create_dataset -e mandrac -n "pool-5_pool_random" -f "(1 + (2 - 0))"

# train on whole idx + RGB, test on whole
python -m detection.merge_index_and_RGB_dataset -d "whole_random" -i "ghost-net-"

# train on pool idx + RGB, test on pool
python -m detection.merge_index_and_RGB_dataset -d "pool_random" -i "pool-"

cd /home/anna/Datasets/created
zip -r datasets.zip  *
scp datasets.zip lariat@10.2.116.180:/home/lariat/code/anna/datasets