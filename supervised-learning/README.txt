# How to run this code

## Download Julia
Using a linux machine, perform the following actions:
1. Download the Julia 1.5 binaries from https://julialang.org/downloads/
    and follow platform specific instructions
    (https://julialang.org/downloads/platform/) to install them.

    on linux, run:
    ```
    cd ~/
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.0-linux-x86_64.tar.gz
    tar -xvzf julia-1.5.0-linux-x86\_64.tar.gz
    rm -r julia-1.5.0-linux-x86\_64.tar.gz
    mkdir ~/.local/bin/
    ln -s ~/julia-1.5.0/bin/julia ~/.local/bin/
    ```
    where `~/.local/bin/` is a dir on your system's $PATH variable. Omit the
    `mkdir ~/.local/bin/` line if you want to put it somewhere else .

## Clone github repository




## Install dependencies


## Run file