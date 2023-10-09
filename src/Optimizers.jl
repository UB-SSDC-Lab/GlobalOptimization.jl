
abstract type Optimizer end

# ===== Interface
optimize!(opt::Optimizer, opts::Options) = _optimize!(opt, opts)
optimize!(opt::Optimizer) = _optimize!(opt, Options())