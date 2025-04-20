import os
import json
from omnigibson.utils.asset_utils import decrypt_file

def decrypt_obj(usd_dir, category, object_id):
    # encrypted_usd file path
    # usd_dir = "real2sim2real_pipeline/deps/OmniGibson/omnigibson/data/og_dataset/objects"

    # category = "straight_chair"
    # object_id = "enuago"
    encrypted_filename = f"{usd_dir}/{category}/{object_id}/usd/{object_id}.encrypted.usd"
    # output usd file path
    usd_path_split = encrypted_filename.split('.')
    usd_path = ".".join([usd_path_split[-3], usd_path_split[-1]])
    # usd_path = f"{usd_dir}/desk/vpwmkm/usd/vpwmkm.usd"
    print(usd_path)

    decrypt_file(encrypted_filename, usd_path)


def decrypt_usd_from_ours_config(scene_info_json_filepath, usd_dir):
    with open(scene_info_json_filepath, 'r') as f:
        scene_info_json_data = json.load(f)

    # Objects
    objects = scene_info_json_data['objects']
    print(objects.keys())

    for obj_name, obj_val in objects.items():
        decrypt_obj(usd_dir=usd_dir, category=obj_val['category'], object_id=obj_val['model'])
        exit()

if __name__ == "__main__":
    scene_info_json_filepath = "<path/to/scene_info.json>"
    usd_base_dir = "<path/to/og_dataset/objects>"

    decrypt_usd_from_ours_config(scene_info_json_filepath=scene_info_json_filepath, usd_dir=usd_base_dir)