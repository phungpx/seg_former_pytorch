import tarfile
import argparse
from pathlib import Path
from urllib import request

# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--url-path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    url_path = Path(args.url_path)
    save_path = data_dir.joinpath(url_path.name)

    if not save_path.exists():
        request.urlretrieve(str(url_path), str(save_path))
        tar = tarfile.TarFile(str(save_path))
        tar.extractall(str(data_dir))
        tar.close()
