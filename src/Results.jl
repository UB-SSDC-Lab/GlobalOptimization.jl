
mutable struct Results{T, XT <: AbstractVector{T}}
    fbest::T 
    xbest::XT

    iters::Int
    time::T 
    exitFlag::Int
end

function Base.show(io::Base.IO, res::Results)
    n = length(res.xbest)

    # Create Format Expressions
    fbestSpec = FormatExpr("f(x) = {1:<e}")
    x1DSpec   = FormatExpr("   x = {1:<e}")
    xnDl1Spec = FormatExpr("   x = [ {1:<e}\t]")
    xnDlmSpec = FormatExpr("       [ {1:<e}\t]")
    itersSpec = FormatExpr("Iters: {1:<d}")
    timeSpec  = FormatExpr("Time:  {1:<e}")

    # Exit flag message
    if res.exitFlag == 1
        exitSpec  = FormatExpr("Maximum reset iterations reached.")
    elseif res.exitFlag == 2
        exitSpec  = FormatExpr("Maximum iterations reached.")
    elseif res.exitFlag == 3
        exitSpec  = FormatExpr("Objective function limit reached.")
    elseif res.exitFlag == 4
        exitSpec  = FormatExpr("Maximum stall time reached.")
    elseif res.exitFlag == 5 
        exitSpec  = FormatExpr("Maximum time reached.")
    end

    # Only print if results have been initialized
    if res.iters != 0
        # Print fbest
        printfmtln(fbestSpec, res.fbest)

        # Print xbest
        if n == 1
            printfmtln(x1DSpec, res.xbest[1])
        else
            printfmtln(xnDl1Spec, res.xbest[1])
            for i in 2:n
                    printfmtln(xnDlmSpec, res.xbest[i])
            end
        end

        # Print iterations and time 
        printfmtln(itersSpec, res.iters)
        printfmtln(timeSpec, res.time)

        # Print exit reason
        printfmtln(exitSpec)
    end
    return nothing
end