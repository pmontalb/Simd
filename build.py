#!/usr/bin/python3
import os
import subprocess
import argparse


def run_command(command):
    try:
        out = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as err:
        print("Err({})\n***stdout({})\n***stderr({})".format(err, err.stdout.decode("UTF-8"), err.stderr.decode("UTF-8")))
        raise err

    def get_non_empty_lines(s):
        return [t.strip() for t in s.decode("UTF-8").split("\n") if t.strip() != ""]
    return get_non_empty_lines(out.stdout), get_non_empty_lines(out.stderr)


def get_presets(cmake_dir):
    presets, _ = run_command("cmake {} --list-presets".format(cmake_dir))
    return [l.strip().replace('"', "") for l in presets if "presets" not in l]


def run_cmake(cmake_dir, preset):
    run_command("cmake {} --preset={}".format(cmake_dir, preset))


def build(cmake_dir, preset):
    run_command("cmake --build {}/cmake-build-{}".format(cmake_dir, preset))


def test(cmake_dir, preset):
    n_cpus = os.cpu_count()
    run_command("ctest --test-dir {}/cmake-build-{} -j {} --output-on-failure".format(cmake_dir, preset, n_cpus))


def run_gcc_coverage(cmake_dir, preset, extra_args):
    run_command("mkdir -p {}/cmake-build-{}/Coverage".format(cmake_dir, preset))

    cwd = os.getcwd()
    os.chdir("{}/cmake-build-{}/Coverage".format(cmake_dir, preset))

    if extra_args != "":
        run_command("gcovr {}/cmake-build-{} -r {} --html-details coverage.html {}".format(cmake_dir, preset, cmake_dir, extra_args))
    else:
        run_command("gcovr {}/cmake-build-{} -r {} --html-details coverage.html".format(cmake_dir, preset, cmake_dir, extra_args))

    os.chdir(cwd)


def run_clang_coverage(cmake_dir, preset, extra_args):
    # TODO
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-dir", "-d", type=str, required=True, help="cmake directory to build")
    parser.add_argument("--debug", "-debug", "-dbg", type=int, help="debug level (0-9)", default=0)
    parser.add_argument("--filter", "-filter", "-f", type=str, help="preset filter", default="")
    parser.add_argument("--ignore", "-ignore", "-i", type=str, help="preset ignore", default="")
    parser.add_argument("--gcovr_extra_args", "-gcovr_extra_args", "-g", type=str, help="extra arguments to pass to gcovr", default="")
    args = parser.parse_args()

    presets = get_presets(args.dir)
    for preset in presets:
        if args.filter != "" and args.filter not in preset:
            continue
        if args.ignore != "" and args.ignore in preset:
            continue

        if args.debug > 0:
            print("running preset {} ...".format(preset))

        if args.debug > 1:
            print("\t[{}] running cmake...".format(preset))
        run_cmake(args.dir, preset)
        if args.debug > 1:
            print("\t[{}] ... cmake done!".format(preset))

        if args.debug > 1:
            print("\t[{}] building...".format(preset))
        build(args.dir, preset)
        if args.debug > 1:
            print("\t[{}] ... build done!".format(preset))

        if args.debug > 1:
            print("\t[{}] testing...".format(preset))
        test(args.dir, preset)
        if args.debug > 1:
            print("\t[{}] ... testing done!".format(preset))

        if preset == "gcc-coverage":
            if args.debug > 1:
                print("\t[{}] running gcovr...".format(preset))
            run_gcc_coverage(args.dir, preset, args.gcovr_extra_args)
            if args.debug > 1:
                print("\t[{}] ... gcovr done!".format(preset))

        if preset == "clang-coverage":
            if args.debug > 1:
                print("\t[{}] running llvm-cov...".format(preset))
            run_clang_coverage(args.dir, preset, None)
            if args.debug > 1:
                print("\t[{}] ... llvm-cov done!".format(preset))

        if args.debug > 0:
            print("... done!")


if __name__ == "__main__":
    main()
