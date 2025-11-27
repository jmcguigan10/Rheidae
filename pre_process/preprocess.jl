using YAML
using HDF5
using Random

include("./get_config.jl")
include("./flux_mod.jl")
include("./residual.jl")
include("./data_mod.jl")


function write_sim_file(output_root::AbstractString,
                        file_cfg::Dict{String,Any},
                        data)
    mkpath(output_root)
    name = get(file_cfg, "output_name", string(file_cfg["name"], "_preprocessed.h5"))
    out_path = joinpath(output_root, name)

    println("  writing per-simulation file: $out_path")

    h5open(out_path, "w") do h5
        write(h5, "F_init",     data.F_init)
        write(h5, "F_true",     data.F_true)
        write(h5, "F_box",      data.F_box)
        write(h5, "F_init_flat", data.F_init_flat)
        write(h5, "F_true_flat", data.F_true_flat)
        write(h5, "F_box_flat",  data.F_box_flat)
        write(h5, "residual",   data.residual)
        write(h5, "invariants", data.invariants)
        write(h5, "input",      data.input)
        write(h5, "target",     data.target)
        write(h5, "sim_id",     data.sim_id)
        write(h5, "dirname",    data.dirname)
    end
end

function main()
    config_path = parse_args()
    cfg = load_config(config_path)

    output_root     = get(cfg, "output_root", ".pdata")
    output_filename = get(cfg, "output_filename", "preprocessed_all.h5")
    mu_split        = get(cfg, "box3d_mu_fraction", 0.5)
    shuffle         = get(cfg, "shuffle", true)
    seed            = get(cfg, "seed", 42)

    files_cfg = cfg["files"]
    @assert !isempty(files_cfg) "config.files is empty"

    Random.seed!(seed)

    datasets = Vector{Any}()

    for file_cfg_any in files_cfg
        file_cfg = Dict{String,Any}(file_cfg_any)
        enabled = get(file_cfg, "enabled", true)
        if !enabled
            println("Skipping disabled file $(file_cfg["name"])")
            continue
        end

        data = process_file(cfg, file_cfg, mu_split)
        write_sim_file(output_root, file_cfg, data)
        push!(datasets, data)
    end

    isempty(datasets) && error("No enabled files were processed; nothing to write.")

    F_init_all      = cat((d.F_init      for d in datasets)...; dims = 1)
    F_true_all      = cat((d.F_true      for d in datasets)...; dims = 1)
    F_box_all       = cat((d.F_box       for d in datasets)...; dims = 1)
    F_init_flat_all = vcat((d.F_init_flat for d in datasets)...)
    F_true_flat_all = vcat((d.F_true_flat for d in datasets)...)
    F_box_flat_all  = vcat((d.F_box_flat  for d in datasets)...)
    input_all       = vcat((d.input       for d in datasets)...)
    target_all      = vcat((d.target      for d in datasets)...)
    residual_all    = cat((d.residual    for d in datasets)...; dims = 1)
    invariants_all  = cat((d.invariants  for d in datasets)...; dims = 1)
    sim_id_all      = vcat((d.sim_id     for d in datasets)...)
    dirname_all     = vcat((d.dirname    for d in datasets)...)

    N_total = size(F_init_all, 1)
    println("------------------------------------------------------------")
    println("Assembled combined dataset with $N_total samples")

    if shuffle
        println("Shuffling combined dataset with seed = $seed")
        perm = randperm(N_total)
        F_init_all     = F_init_all[perm, :, :]
        F_true_all     = F_true_all[perm, :, :]
        F_box_all      = F_box_all[perm, :, :]
        F_init_flat_all = F_init_flat_all[perm, :]
        F_true_flat_all = F_true_flat_all[perm, :]
        F_box_flat_all  = F_box_flat_all[perm, :]
        input_all       = input_all[perm, :]
        target_all      = target_all[perm, :]
        residual_all   = residual_all[perm, :]
        invariants_all = invariants_all[perm, :]
        sim_id_all     = sim_id_all[perm]
        dirname_all    = dirname_all[perm]
    else
        println("Shuffling disabled")
    end

    mkpath(output_root)
    out_path = joinpath(output_root, output_filename)

    println("Writing combined dataset to $out_path")

    h5open(out_path, "w") do h5
        write(h5, "F_init",     F_init_all)
        write(h5, "F_true",     F_true_all)
        write(h5, "F_box",      F_box_all)
        write(h5, "F_init_flat", F_init_flat_all)
        write(h5, "F_true_flat", F_true_flat_all)
        write(h5, "F_box_flat",  F_box_flat_all)
        write(h5, "input",      input_all)
        write(h5, "target",     target_all)
        write(h5, "residual",   residual_all)
        write(h5, "invariants", invariants_all)
        write(h5, "sim_id",     sim_id_all)
        write(h5, "dirname",    dirname_all)
    end

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
