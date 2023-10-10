function exprate(λ)
    return -log(1.0 - rand()) / λ
end
function expscale(β)
    return -β*log(1.0 - rand())
end

function laplace(μ, b)
    u = rand() - 0.5
    return μ - b*sign(u)*log(1.0 - 2.0*abs(u))
end
laplace(b) = laplace(0.0, b)
laplace() = laplace(0.0, 1.0)