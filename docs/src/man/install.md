# Installation Guide

This installation guide details two different approaches to installing GlobalOptimization.jl. The first is the simplest approach, but does assume you're simply installing GlobalOptimization.jl to use it elsewhere and do not intend on modifying the GlobalOptimization.jl source code. The second, involves more steps, but will install GlobalOptimization.jl in `dev` mode, which allows you to modify the source code (and use the modified code directly).

## Simple Install (No Development)

For the simple installation approach, we simply need to employ the standard approach to adding a new Julia package as a dependency in an environment, but we will need to use the repository URL instead of the package name. Therefore, to install GlobalOptimization.jl into the desired Julia environment, open a Julia REPL and activate the environment you'd like to install GlobalOptimizaiton.jl into.

This can be achieved with the following commands: