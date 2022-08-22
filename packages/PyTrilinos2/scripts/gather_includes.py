import glob
import os

def get_angular_include(line):
    first=True
    for i in range(len(line)):
        if line[i] == '"':
            if first:
                line = line[:i] + '<' + line[i+1:]
                first = False
            else:
                line = line[:i] + '>' + line[i+1:]
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

def copy_and_angular_includes(filenames, from_dir, to_dir):
    output_filenames = []
    # loops over the files, replace include " " by include < > and write them in the to_dir:
    for filename in filenames:
        file_name, file_extension = os.path.splitext(filename)
        if file_extension == '.cpp':
            write_extension = '_cpp.hpp'
        else:
            write_extension = file_extension
        output_filenames.append(file_name+write_extension)
        with open(from_dir+'/'+filename, 'r') as from_f:
            with open(to_dir+'/'+file_name+write_extension, 'w') as to_f:
                for line in from_f:
                    if line.startswith('#include'):
                        line = get_angular_include(line)
                    to_f.write(f'{line}')
    return output_filenames

if __name__ == '__main__':

    cwd = os.getcwd()
    trilinos_include = '/home/knliege/local/trilinos/albany_release/include'
    tpetra_src = '/home/knliege/dev/trilinos/Trilinos/packages/tpetra/core/src'
    tpetra_build = '/home/knliege/dev/trilinos/Trilinos_Release_B/packages/tpetra/core/src'

    teuchos_include_list = 'Teuchos_header_list.txt'
    tpetra_include_list = 'All_header_list.txt'
    tpetra_src_list = 'Tpetra_src_list.txt'
    tpetra_build_list = 'Tpetra_build_list.txt'

    with open(teuchos_include_list, 'r') as fh:
        teuchos_include_filenames = fh.read().splitlines()

    copy_and_angular_includes(['Teuchos_Include.hpp'], cwd+'/include', cwd+'/include_teuchos_tmp')
    teuchos_include_filenames = copy_and_angular_includes(teuchos_include_filenames, trilinos_include, cwd+'/include_teuchos_tmp')
    
    with open(tpetra_include_list, 'r') as fh:
        tpetra_include_filenames = fh.read().splitlines()

    copy_and_angular_includes(['Teuchos_Include.hpp'], cwd+'/include', cwd+'/include_tpetra_tmp')
    copy_and_angular_includes(['Tpetra_Include.hpp'], cwd+'/include', cwd+'/include_tpetra_tmp')
    copy_and_angular_includes(tpetra_include_filenames, trilinos_include, cwd+'/include_tpetra_tmp')

    make_all_includes_from_filenames('teuchos_includes.hpp', [cwd+'/include/Teuchos_Binder.hpp'])
    make_all_includes_from_filenames('tpetra_includes.hpp', [cwd+'/include/Tpetra_Binder.hpp'])
