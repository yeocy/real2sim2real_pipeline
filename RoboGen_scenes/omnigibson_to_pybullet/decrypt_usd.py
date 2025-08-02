import os
import json
from omnigibson.utils.asset_utils import decrypt_file

def decrypt_obj(usd_dir, category, object_id):
    # encrypted_usd file path
    # usd_dir = "real2sim2real_pipeline/deps/OmniGibson/omnigibson/data/og_dataset/objects"

    # category = "straight_chair"
    # object_id = "enuago"
    print(f"[{category}]")
    encrypted_filename = f"{usd_dir}/{category}/{object_id}/usd/{object_id}.encrypted.usd"
    # output usd file path
    usd_path_split = encrypted_filename.split('.')
    usd_path = ".".join([usd_path_split[-3], usd_path_split[-1]])
    # usd_path = f"{usd_dir}/desk/vpwmkm/usd/vpwmkm.usd"
    decrypt_file(encrypted_filename, usd_path)

    print(f"decrypted usd: {usd_path}")


def urdf_exists(urdf_base_dir, category, object_id):
    # Check if the decrypted URDF file exists
    urdf_path = f"{urdf_base_dir}/{category}/{object_id}/{object_id}.urdf"
    if os.path.exists(urdf_path):
        is_urdf_exists = True
    else:
        is_urdf_exists = False

    return os.path.dirname(urdf_path), is_urdf_exists

def decrypt_usd_from_ours_config(scene_info_json_filepath, usd_dir, urdf_base_dir):
    with open(scene_info_json_filepath, 'r') as f:
        scene_info_json_data = json.load(f)

    # Objects
    objects = scene_info_json_data['objects']
    print(objects.keys())

    for obj_name, obj_val in objects.items():
        decrypt_obj(usd_dir=usd_dir, category=obj_val['category'], object_id=obj_val['model'])

        urdf_path, is_urdf_exists = urdf_exists(urdf_base_dir=urdf_base_dir, category=obj_val['category'], object_id=obj_val['model'])
        if is_urdf_exists:
            print(f"{urdf_path}")
        else:
            print(f"* {urdf_path}")

if __name__ == "__main__":
    # scene_info_json_filepath = "<path/to/scene_info.json>"
    # usd_base_dir = "<path/to/og_dataset/objects>"

    # scene_info_json_filepath = "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/maniskill_hab/acdc_output/task_scene_generation/scene_info.json"
    usd_base_dir = "/home/kodogyu/github_repos/digital-cousins/deps/OmniGibson/omnigibson/data/og_dataset/objects"
    # urdf_base_dir = "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/urdf"

    # decrypt_usd_from_ours_config(scene_info_json_filepath=scene_info_json_filepath, usd_dir=usd_base_dir, urdf_base_dir=urdf_base_dir)

    # category = "fridge"
    # object_ids = ["dszchb", "dxwbae", "hivvdf", "hzgqdn", "jtqazu", "juwaoh", "lleghp", "petcxr", "seyhuo", "vwylob", "xyejdx"]
    category = "folder"
    object_ids = ["enzfco"]
    for object_id in object_ids:
        decrypt_obj(usd_dir=usd_base_dir, category=category, object_id=object_id)