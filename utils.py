import tarfile
import os

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


def extractTARS(root):
    subjects = []
    for tarPath in find_files_with_extension(root, '.tar'):
        with tarfile.open(tarPath, 'r') as tar:
            tar.extractall(path=root)

    for file in find_files_with_extension(root, '.edf'):
        subjects.append(int(file[file.rfind(".")-1]))
    subjects.sort()
    return subjects
