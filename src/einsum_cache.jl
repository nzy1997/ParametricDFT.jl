# ============================================================================
# Einsum Contraction Code Cache
# ============================================================================
# Caches optimized OMEinsum contraction codes to disk so that expensive
# TreeSA optimization runs only once per unique (einsum_code, size_dict)
# combination. Subsequent calls with the same inputs load from cache.
# ============================================================================

using Serialization
using SHA

const EINSUM_CACHE_DIR = Ref(joinpath(homedir(), ".cache", "ParametricDFT", "einsum_codes"))

"""
    set_einsum_cache_dir!(dir::String)

Set the directory used for caching optimized einsum codes.
"""
function set_einsum_cache_dir!(dir::String)
    EINSUM_CACHE_DIR[] = dir
end

"""
    _cache_key(flat_code, size_dict)

Compute a stable hash key for an einsum code + size dict combination.
"""
function _cache_key(flat_code, size_dict)
    # Build a canonical string representation
    ixs = getixsv(flat_code)
    iy = getiyv(flat_code)
    repr_str = string("ixs=", ixs, ";iy=", iy, ";sizes=", sort(collect(size_dict)))
    return bytes2hex(sha256(repr_str))
end

"""
    optimize_code_cached(flat_code, size_dict, optimizer=TreeSA())

Like `optimize_code(flat_code, size_dict, optimizer)` but caches the result
to disk. On cache hit, returns immediately without running the optimizer.
"""
function optimize_code_cached(flat_code, size_dict, optimizer=TreeSA())
    cache_dir = EINSUM_CACHE_DIR[]
    key = _cache_key(flat_code, size_dict)
    cache_path = joinpath(cache_dir, "$(key).jls")

    # Cache hit
    if isfile(cache_path)
        try
            result = deserialize(cache_path)
            @info "Loaded cached einsum code" key=key[1:12]
            return result
        catch e
            @warn "Cache read failed, recomputing" exception=e
            rm(cache_path; force=true)
        end
    end

    # Cache miss — run optimizer
    @info "Optimizing einsum contraction (this may take a few minutes for large circuits)..."
    result = optimize_code(flat_code, size_dict, optimizer)

    # Save to cache
    mkpath(cache_dir)
    try
        serialize(cache_path, result)
        @info "Cached einsum code" key=key[1:12]
    catch e
        @warn "Cache write failed" exception=e
    end

    return result
end
