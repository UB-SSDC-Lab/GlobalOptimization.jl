
# Developing Documentation
When developing documentation locally, it is suggested to use `servedocs()` provided by
[LiveServer.jl](https://github.com/JuliaDocs/LiveServer.jl) to tests the documentation 
build process while viewing updates to them dynamically as they are made. This can be 
done by running the following command in your terminal while at the base level of your
local instance of GlobalOptimization.jl:

```
julia --project=docs -ie 'using GlobalOptimization, LiveServer; servedocs(include_dirs=["src/"])'
```

If the documentation build is successful, this will print a link to a spawned local server
that you can open in any browser.