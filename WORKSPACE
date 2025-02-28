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

new_local_repository(
    name = "cppad",
    path = "/usr/local",  # Path where CppAD is installed
    build_file_content = """
    cc_library(
        name = "cppad",
        hdrs = glob(["include/cppad/**/*.hpp"]),
        includes = ["include"],
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "cppad_lib",
        srcs = ["lib/libcppad_lib.so"],  # Use .so if it's shared
        includes = ["include"],
        visibility = ["//visibility:public"],
    )
    """,
)
