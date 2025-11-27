"""
Helpers for reading raw flux HDF5 files and building training-ready arrays.
"""

"""
Flatten (N, 6, 4) flux arrays to (N, 24) with species-major ordering.
"""
function flatten_flux(F::Array{<:Real,3})
    N, nspecies, ncomp = size(F)
    @assert nspecies == 6 "expected 6 species"
    @assert ncomp == 4 "expected 4 components"
    flat = Array{Float32}(undef, N, nspecies * ncomp)
    @inbounds for k in 1:N
        idx = 1
        for a in 1:nspecies
            for μ in 1:ncomp
                flat[k, idx] = Float32(F[k, a, μ])
                idx += 1
            end
        end
    end
    return flat
end

"""
Compute simple invariants from initial flux:
for each species, store density (t component) and spatial magnitude.
Returns (N, 12).
"""
function compute_invariants(F_init::Array{<:Real,3})
    N, nspecies, ncomp = size(F_init)
    @assert nspecies == 6 "expected 6 species"
    @assert ncomp == 4 "expected 4 components"

    invariants = Array{Float32}(undef, N, nspecies * 2)
    @inbounds for k in 1:N
        idx = 1
        for a in 1:nspecies
            Fx, Fy, Fz = F_init[k, a, 1], F_init[k, a, 2], F_init[k, a, 3]
            density = F_init[k, a, 4]
            invariants[k, idx] = Float32(density)
            invariants[k, idx + 1] = Float32(sqrt(Fx * Fx + Fy * Fy + Fz * Fz))
            idx += 2
        end
    end
    return invariants
end

"""
Process a single Emu_data HDF5 file according to config entry.
Returns a NamedTuple with all arrays for that file.
"""
function process_file(cfg, file_cfg::Dict{String,Any}, mu_split::Real)
    data_root = get(cfg, "data_root", ".data")
    h5_paths = cfg["h5_paths"]
    lebedev_order = get(cfg, "lebedev_order", 13)

    rel_path = file_cfg["path"]
    sim_path = joinpath(data_root, rel_path)
    sim_id = String(file_cfg["sim_id"])
    max_rows = get(file_cfg, "max_rows", nothing)

    println("------------------------------------------------------------")
    println("Processing file: $sim_path")
    println("  sim_id   = $sim_id")
    println("  max_rows = $(max_rows === nothing ? "all" : string(max_rows))")

    F4_init = nothing
    F4_final = nothing
    dirnames = nothing

    h5open(sim_path, "r") do h5
        F4_init  = read(h5[h5_paths["F_init"]])
        F4_final = read(h5[h5_paths["F_true"]])

        if haskey(h5_paths, "dirnames") && haskey(h5, h5_paths["dirnames"])
            dirnames = read(h5[h5_paths["dirnames"]])
        end
    end

    # Convert to (N, 6, 4)
    F_init6 = convert_F4_to_species(F4_init)
    F_true6 = convert_F4_to_species(F4_final)
    N_total = size(F_init6, 1)
    @assert size(F_true6, 1) == N_total "F4_initial and F4_final must have same number of samples"

    if dirnames !== nothing
        dirnames = vec(String.(dirnames))
    end

    if max_rows !== nothing && max_rows > 0 && max_rows < N_total
        N = max_rows
        F_init6 = F_init6[1:N, :, :]
        F_true6 = F_true6[1:N, :, :]
        if dirnames !== nothing && length(dirnames) >= N
            dirnames = dirnames[1:N]
        end
        println("  truncated to first $N rows")
    else
        N = N_total
        if dirnames !== nothing && length(dirnames) != N
            dirnames = length(dirnames) >= N ? dirnames[1:N] : fill("", N)
        end
        println("  using all $N rows")
    end

    # Use Box3D flux function to compute predicted final flux
    F_box6 = box3d_flux_on_six(F_init6; mu_split=mu_split, lebedev_order=lebedev_order)

    residual   = compute_residual(F_true6, F_box6)
    invariants = compute_invariants(F_init6)

    sim_ids = fill(sim_id, N)

    # directorynames: keep for debugging, but not required for training
    dirname_vec =
        dirnames === nothing ? fill("", N) : String.(dirnames)

    F_init32 = Float32.(F_init6)
    F_true32 = Float32.(F_true6)
    F_box32  = Float32.(F_box6)
    F_init_flat = flatten_flux(F_init32)
    F_true_flat = flatten_flux(F_true32)
    F_box_flat  = flatten_flux(F_box32)
    input = hcat(F_init_flat, F_box_flat)  # shape (N, 48)
    target = F_true_flat                   # shape (N, 24)

    return (F_init = F_init32,
            F_true = F_true32,
            F_box = F_box32,
            F_init_flat = F_init_flat,
            F_true_flat = F_true_flat,
            F_box_flat = F_box_flat,
            input = input,
            target = target,
            residual = residual,
            invariants = invariants,
            sim_id = sim_ids,
            dirname = dirname_vec)
end
