"""
usd_portable.py — 풀 USD 의 텍스처 참조를 상대경로(./textures/...)로 바꿔 풀 폴더를 옮겨도 쓰게 한다.

build_og_usd.py 는 텍스처를 <OG_DATASET>/objects/<cat>/<model>/usd/textures/ 절대경로로 쓴다
(OG 가 암호화 USD 를 임시 폴더에 풀어서 열기 때문). 풀에는 평문 <model>.usd 를 이 함수로 상대경로로
고쳐 두고, asset_pool.use_pool_usd 가 그 평문 USD 를 제자리에서 열게 한다.

pxr 이 필요하다: Isaac(OmniGibson) 을 og.launch() 한 뒤에 부른다.
"""
import os
import shutil


def make_portable_usd(src_usd, dst_dir, model):
    """src_usd 를 dst_dir/<model>.usd 로 저장하면서 텍스처를 ./textures/<파일> 로 바꾸고 텍스처도 복사한다.

    Returns:
        list of str: 바꾼 텍스처 파일 이름
    """
    from pxr import Usd, UsdShade, Sdf

    os.makedirs(os.path.join(dst_dir, "textures"), exist_ok=True)
    stage = Usd.Stage.Open(src_usd)
    changed = []
    for prim in stage.Traverse():
        shader = UsdShade.Shader(prim)
        if not shader or shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        inp = shader.GetInput("file")
        val = inp.Get() if inp else None
        if val is None:
            continue
        path = val.path if hasattr(val, "path") else str(val)
        name = os.path.basename(path)
        src_tex = path if os.path.isabs(path) and os.path.exists(path) else \
            os.path.join(os.path.dirname(src_usd), "textures", name)
        if os.path.exists(src_tex) and os.path.abspath(src_tex) != os.path.abspath(
                os.path.join(dst_dir, "textures", name)):
            shutil.copy2(src_tex, os.path.join(dst_dir, "textures", name))
        inp.Set(Sdf.AssetPath(f"./textures/{name}"))
        changed.append(name)
    stage.GetRootLayer().Export(os.path.join(dst_dir, f"{model}.usd"))
    return changed
