import os
import shutil
from pathlib import Path
from typing import Dict, List, TypedDict
from zipfile import ZipFile
import requests
from PIL import Image
from rich.progress import BarColumn, DownloadColumn, MofNCompleteColumn, Progress, TextColumn, TransferSpeedColumn
from tqdm import tqdm
from prismatic.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)
DatasetComponent = TypedDict('DatasetComponent', {'name': str, 'extract': bool, 'extract_type': str, 'url': str, 'do_rename': bool}, total=False)
DATASET_REGISTRY: Dict[str, List[DatasetComponent]] = {'llava-laion-cc-sbu-558k': [{'name': 'chat.json', 'extract': False, 'url': 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json', 'do_rename': True}, {'name': 'images', 'extract': True, 'extract_type': 'directory', 'url': 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip', 'do_rename': False}], 'llava-v1.5-instruct': [{'name': 'llava_v1_5_mix665k.json', 'extract': False, 'url': 'https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json', 'do_rename': True}, {'name': 'coco/train2017', 'extract': True, 'extract_type': 'directory', 'url': 'http://images.cocodataset.org/zips/train2017.zip', 'do_rename': True}, {'name': 'gqa/images', 'extract': True, 'extract_type': 'directory', 'url': 'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip', 'do_rename': True}, {'name': 'ocr_vqa/images', 'extract': True, 'extract_type': 'directory', 'url': 'https://huggingface.co/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip', 'do_rename': True}, {'name': 'textvqa/train_images', 'extract': True, 'extract_type': 'directory', 'url': 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip', 'do_rename': True}, {'name': 'vg/VG_100K', 'extract': True, 'extract_type': 'directory', 'url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip', 'do_rename': True}, {'name': 'vg/VG_100K_2', 'extract': True, 'extract_type': 'directory', 'url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip', 'do_rename': True}]}

def convert_to_jpg(image_dir: Path) -> None:
    overwatch.info(f'Converting all Images in `{image_dir}` to JPG')
    for image_fn in tqdm(list(image_dir.iterdir())):
        if image_fn.suffix in {'.jpg', '.jpeg'} or (jpg_fn := (image_dir / f'{image_fn.stem}.jpg')).exists():
            continue
        if image_fn.suffix == '.gif':
            gif = Image.open(image_fn)
            gif.seek(0)
            gif.convert('RGB').save(jpg_fn)
        elif image_fn.suffix == '.png':
            Image.open(image_fn).convert('RGB').save(jpg_fn)
        else:
            raise ValueError(f'Unexpected image format `{image_fn.suffix}`')

def download_with_progress(url: str, download_dir: Path, chunk_size_bytes: int=1024) -> Path:
    overwatch.info(f'Downloading {(dest_path := (download_dir / Path(url).name))} from `{url}`', ctx_level=1)
    if dest_path.exists():
        return dest_path
    response = requests.get(url, stream=True)
    with Progress(TextColumn('[bold]{task.description} - {task.fields[fname]}'), BarColumn(bar_width=None), '[progress.percentage]{task.percentage:>3.1f}%', '•', DownloadColumn(), '•', TransferSpeedColumn(), transient=True) as dl_progress:
        dl_tid = dl_progress.add_task('Downloading', fname=dest_path.name, total=int(response.headers.get('content-length', 'None')))
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(chunk_size=chunk_size_bytes):
                dl_progress.advance(dl_tid, f.write(data))
    return dest_path

def extract_with_progress(archive_path: Path, download_dir: Path, extract_type: str, cleanup: bool=False) -> Path:
    assert archive_path.suffix == '.zip', 'Only `.zip` compressed archives are supported for now!'
    overwatch.info(f'Extracting {archive_path.name} to `{download_dir}`', ctx_level=1)
    with Progress(TextColumn('[bold]{task.description} - {task.fields[aname]}'), BarColumn(bar_width=None), '[progress.percentage]{task.percentage:>3.1f}%', '•', MofNCompleteColumn(), transient=True) as ext_progress:
        with ZipFile(archive_path) as zf:
            ext_tid = ext_progress.add_task('Extracting', aname=archive_path.name, total=len((members := zf.infolist())))
            extract_path = Path(zf.extract(members[0], download_dir))
            if extract_type == 'file':
                assert len(members) == 1, f'Archive `{archive_path}` with extract type `{extract_type} has > 1 member!'
            elif extract_type == 'directory':
                for member in members[1:]:
                    zf.extract(member, download_dir)
                    ext_progress.advance(ext_tid)
            else:
                raise ValueError(f'Extract type `{extract_type}` for archive `{archive_path}` is not defined!')
    if cleanup:
        archive_path.unlink()
    return extract_path

def download_extract(dataset_id: str, root_dir: Path) -> None:
    os.makedirs((download_dir := (root_dir / 'download' / dataset_id)), exist_ok=True)
    dl_tasks = [d for d in DATASET_REGISTRY[dataset_id] if not (download_dir / d['name']).exists()]
    for dl_task in dl_tasks:
        dl_path = download_with_progress(dl_task['url'], download_dir)
        if dl_task['extract']:
            dl_path = extract_with_progress(dl_path, download_dir, dl_task['extract_type'])
            dl_path = dl_path.parent if dl_path.is_file() else dl_path
        if dl_task['do_rename']:
            shutil.move(dl_path, download_dir / dl_task['name'])
