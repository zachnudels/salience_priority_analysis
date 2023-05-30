import tarfile
import os
import zipfile


def make_dir(path):
    try:
        os.mkdir(path)
        print(f"Directory created successfully: {path}")
    except FileExistsError:
        print(f"Directory already exists: {path}")
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")


def find_files_with_extension(root_dir, extension):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths


def unzipData(root):
    subjects = []
    tarPath = find_files_with_extension(root, '.tar')
    if len(tarPath) > 0:
        with tarfile.open(tarPath[0], 'r') as tar:
            tar.extractall(path=root)
    else:
        zipPath = find_files_with_extension(root, '.zip')
        with zipfile.ZipFile(zipPath[0], 'r') as zip_ref:
            zip_ref.extractall(path=root)

    for file in find_files_with_extension(root, '.edf'):
        subjects.append(int(file[file.rfind(".")-1]))
    subjects.sort()
    return subjects
