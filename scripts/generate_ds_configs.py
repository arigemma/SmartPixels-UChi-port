from glob import glob
import yaml
from pathlib import Path


ds_to_path = {
    "barrel-phys-pt": "dataset_2s",
    "barrel-flat-pt": "dataset_3s",
    "endcap-phys-pt": "dataset_6s",
    "endcap-flat-pt": "dataset_7s",
    "irradiated-barrel-phys-pt": "dataset_8s",
    "irradiated-barrel-flat-pt": "dataset_9s",
}

base_path = "/scratch/gpfs/IOJALVO/lg0508/SmartPixels/17180303"

for ds_name, ds_folder in ds_to_path.items():
    ds_path = base_path + f"/{ds_folder}/"
    for sub_folder in glob(ds_path + "*/"):
        sub_name = sub_folder.split("/")[-2].split("_")[-1]
        label_files = sorted(glob(sub_folder + "label*.parquet"))
        reco_files = sorted(glob(sub_folder + "reco*.parquet"))
        n_test = len(label_files) // 3
        label_train_files = label_files[:-n_test]
        label_test_files = label_files[-n_test:]
        reco_train_files = reco_files[:-n_test]
        reco_test_files = reco_files[-n_test:]

        # Extract just filenames, not full paths
        file_dict = {
            "root": sub_folder,
            "train": [[Path(r).name, Path(l).name] for r, l in zip(reco_train_files, label_train_files)],
            "test": [[Path(r).name, Path(l).name] for r, l in zip(reco_test_files, label_test_files)],
        }
        # save file_dict as yaml
        # create dict <ds_name> if doesnt exist
        import os
        if not os.path.exists(ds_name):
            os.makedirs(ds_name) 

        with open(ds_name + "/" + sub_name + ".yml", "w") as f:
            f.write(f"root: {sub_folder}\n")
            f.write("train:\n")
            for r, l in zip(reco_train_files, label_train_files):
                f.write(f"- [{Path(r).name}, {Path(l).name}]\n")
            f.write("test:\n")
            for r, l in zip(reco_test_files, label_test_files):
                f.write(f"- [{Path(r).name}, {Path(l).name}]\n")            
        # with open(ds_name + "/" + sub_name + ".yml", "w") as f:
        #     yaml.dump(file_dict, f)
        # ds_files = glob(sub_folder + "*.parquet")
        # print(f"Dataset: {ds_name}, Sub-folder: {sub_folder}, n label files : {len(label_files)}, n reco files : {len(reco_files)}, n test files : {n_test}")
    # print("\n")
