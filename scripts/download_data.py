from pathlib import Path
import urllib.request
import zipfile

BASE_URL = "https://zenodo.org/records/5706578/files/{name}?download=1"
FILES = ["Train.zip", "Val.zip"]
CHUNK = 1024 * 1024 #1 MB
OUT = Path("data/raw")


def download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            print(f"\r {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="")
    print()
    tmp.rename(dest)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        archive = OUT / name
        if archive.exists():
            print(f"{name}: already downloaded")
        else:
            print(f"{name}: downloading")
            download(BASE_URL.format(name=name), archive)


        marker = OUT / f".{name}.extracted"
        if not marker.exists():
            print(f"{name}: extracting")
            with zipfile.ZipFile(archive) as z:
                z.extractall(OUT)
                marker.touch()


if __name__ == "__main__":
    main()

