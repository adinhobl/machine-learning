# How to run this project's code

1. Set Up Julia
    Download the Julia 1.5 binaries from https://julialang.org/downloads/
    and follow platform specific instructions
    (https://julialang.org/downloads/platform/) to install them.

    On linux, run:
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

2. Clone github repository
    Download the project from gitlab (https://github.com/adinhobl/Machine-Learning)
    On linux, run:
    ```
    git clone https://github.com/adinhobl/machine-learning.git
    cd machine-learning/supervised-learning
    ```
    from the command line. 

3. Install dependencies
    Enter Julia prompt by typing:
    ```
    julia
    ```
    at the command line. If it is on your $PATH, it should run. 
    Next type:
    ```
    ] instantiate
    ```
    This should load all the project's packages from the Project.toml and
    Manifest.toml files. Hit backspace to get back to the Julia prompt from the Pkg screen

4. Make sure Scikit-Learn is installed with conda (or pip) and they are usable in python
    I did this from the command line:
    ```
    conda install scikit-learn
    ```

5. Make sure PyCall works to interface with SKLearn
    Next type this in the julia prompt:
    ```
    using Pkg
    Pkg.build("PyCall")
    ```

6. To run the jupyter notebook, you will need jupyter installed on your current path, either in a conda environment, or locally.
    Alternatively, Julia can download it if you run:
    ```
    ] build IJulia
    ``` 
    from the Julia prompt. Then navigate to the appropriate directory, and run
    ```
    using IJulia
    notebook(dir=pwd())
    ```

7. 