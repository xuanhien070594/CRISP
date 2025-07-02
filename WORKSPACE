# -*- mode: python -*-
# vi: set ft=python :
workspace(name = "CRISP")
DRAKE_COMMIT = "v1.28.0"

DRAKE_CHECKSUM = "6ff298d7fbc33cb17963509f86fcd9cb6816d455b97b3fd589e1085e0548c2fe"
# Before changing the COMMIT, temporarily uncomment the next line so that Bazel
# displays the suggested new value for the CHECKSUM.
# DRAKE_CHECKSUM = "0" * 64

# Maybe download Drake.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "drake",
    sha256 = DRAKE_CHECKSUM,
    strip_prefix = "drake-{}".format(DRAKE_COMMIT.strip("v")),
    urls = [x.format(DRAKE_COMMIT) for x in [
        "https://github.com/RobotLocomotion/drake/archive/{}.tar.gz",
    ]],
)

# Reference external software libraries and tools per Drake's defaults.  Some
# software will come from the host system (Ubuntu or macOS); other software
# will be downloaded in source or binary form from github or other sites.
load("@drake//tools/workspace:default.bzl", "add_default_workspace")

add_default_workspace()

# Assume CppAD and PIQP are already installed in /usr/local, we can just use the existing headers and library
new_local_repository(
    name = "cppad",
    path = "/usr/local",  # Path where CppAD is installed
    build_file_content = """
cc_library(
    name = "cppad_shared_lib",
    srcs = ["lib/libcppad_lib.so"],  # Use .so if it's shared
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "piqp",
    path = "/usr/local",  # Path where PIQP is installed
    build_file_content = """
cc_library(
    name = "piqp_shared_lib",
    srcs = ["lib/libpiqp.so"],  # Use .so if it's shared
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    url = "https://github.com/mikael-s-persson/bazel-compile-commands-extractor/archive/f5fbd4cee671d8d908f37c83abaf70fba5928fc7.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-f5fbd4cee671d8d908f37c83abaf70fba5928fc7",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
hedron_compile_commands_setup_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive_transitive()
