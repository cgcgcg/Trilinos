import glob
import os
import sys

def get_without_subfolder(line):
    last_index=-1
    for i in range(len(line)):
        if line[i] == '/':
            last_index = i
        if line[i] == '<':
            first_index = i
    if last_index == -1:
        return line
    return line[:first_index+1]+line[last_index+1:]

def get_angular_include(line, remove_subfolder=False):
    first=True
    for i in range(len(line)):
        if line[i] == '"':
            if first:
                line = line[:i] + '<' + line[i+1:]
                first = False
            else:
                line = line[:i] + '>' + line[i+1:]
    if remove_subfolder:
        return get_without_subfolder(line)
    return line

def make_all_includes(all_include_filename, folders):
    all_includes = []
    for folder in folders:
        for filename in (glob.glob(f'{folder}/**/*.hpp', recursive=True) +
                        glob.glob(f'{folder}/**/*.cpp', recursive=True) +
                        glob.glob(f'{folder}/**/*.h', recursive=True) +
                        glob.glob(f'{folder}/**/*.cc', recursive=True) +
                        glob.glob(f'{folder}/**/*.c', recursive=True)):
            with open(filename, 'r') as fh:
                for line in fh:
                    if line.startswith('#include'):
                        all_includes.append(get_angular_include(line).strip())
    all_includes = list(set(all_includes))
    # This is to ensure that the list is always the same and doesn't
    # depend on the filesystem state.  Not technically necessary, but
    # will cause inconsistent errors without it.
    all_includes.sort()
    with open(all_include_filename, 'w') as fh:
        for include in all_includes:
            fh.write(f'{include}\n')
    return all_include_filename

def make_all_includes_from_filenames(all_include_filename, filenames):
    all_includes = []
    for filename in filenames:
        with open(filename, 'r') as fh:
            for line in fh:
                if line.startswith('#include'):
                    all_includes.append(get_angular_include(line).strip())
    all_includes = list(set(all_includes))
    # This is to ensure that the list is always the same and doesn't
    # depend on the filesystem state.  Not technically necessary, but
    # will cause inconsistent errors without it.
    all_includes.sort()
    with open(all_include_filename, 'w') as fh:
        for include in all_includes:
            fh.write(f'{include}\n')
    return all_include_filename

# https://github.com/RosettaCommons/binder/issues/212

def copy_and_angular_includes(filenames, to_dir):
    output_filenames = []
    # loops over the files, replace include " " by include < > and write them in the to_dir:
    for filename in filenames:
        file_name, file_extension = os.path.splitext(os.path.basename(filename))
        if file_extension == '.cpp':
            write_extension = '_cpp.hpp'
        else:
            write_extension = file_extension
        output_filenames.append(file_name+write_extension)
        with open(filename, 'r') as from_f:
            with open(to_dir+'/'+file_name+write_extension, 'w') as to_f:
                for line in from_f:
                    if line.startswith('#include'):
                        line = get_angular_include(line, True)
                    to_f.write(f'{line}')
    return output_filenames

if __name__ == '__main__':
    CMAKE_CURRENT_SOURCE_DIR = sys.argv[1]
    CMAKE_CURRENT_BINARY_DIR = sys.argv[2]
    all_header_list = sys.argv[3]
    binder_include_name = sys.argv[4]

    with open(all_header_list, 'r') as fh:
        all_include_filenames = fh.read().splitlines()

    copy_and_angular_includes(all_include_filenames, CMAKE_CURRENT_BINARY_DIR+'/include_tmp')
    make_all_includes_from_filenames(binder_include_name, [CMAKE_CURRENT_SOURCE_DIR+'/src/PyTrilinos2_Binder_Input.hpp'])
