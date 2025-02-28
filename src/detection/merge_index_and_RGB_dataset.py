import os
import shutil
import click

@click.command()
@click.option("--dataset_type", "-d", help="Dataset type (e.g. _whole_random)")
@click.option("-index_dataset", "-i", default="", help="Index dataset name (e.g. ghost-net-)")

def main(dataset_type, index_dataset):
    """
    Combine the index dataset with the RGB train set. The val and test sets remain the same.
    e.g. from ghost-net-whole_random and RGB_whole_random to ghost-net-RGB-and-whole_random

    Args:
    dataset_type (str): Train/test dataset type (e.g. _whole_random)
    index_dataset (str): Name of dataset used for creating the index (e.g. ghost-net-)

    Returns:
    None
    """

    dataset_path = "/home/anna/Datasets/created"
    rgb_path = f"{dataset_path}/RGB_{dataset_type}"
    dirs = next(os.walk(dataset_path))[1]
    for dir in dirs:
        if dataset_type in dir and index_dataset in dir and dir != "RGB_{dataset_type}":
            new_name = f"{dir.split('_')[0]}_RGB-and-{'_'.join(dir.split('_')[1:])}"
            print(dir, " -> ", new_name)
            shutil.copytree(f'{dataset_path}/{dir}', f'{dataset_path}/{new_name}')
            for item in ("images", "labels"):
                for file in os.listdir(f'{dataset_path}/{new_name}/{item}/train'):
                    RGB_file_name = f"{'_'.join(file.split('_')[:-1])}_RGB.{file.split('.')[-1]}"
                    shutil.copy2(f'{rgb_path}/{item}/train/{RGB_file_name}', f'{dataset_path}/{new_name}/{item}/train/')

if __name__ == "__main__":
    main()

            

                    
            
