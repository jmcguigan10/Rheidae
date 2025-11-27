
include("./Box3D.jl")

function convert_F4_to_species(F4::Array{<:Real,4})
    dims = size(F4)

    # Find which dim is which
    idx_comp = findfirst(==(4), dims)   # spacetime components
    idx_nnu  = findfirst(==(2), dims)   # nu / antinu
    idx_flav = findfirst(==(3), dims)   # flavor
    idx_species = findfirst(==(6), dims)

    function sample_idx(used)
        used_vec = collect(used)
        idxs = setdiff(collect(1:length(dims)), used_vec)
        length(idxs) == 1 || error("could not infer sample dimension from shape $(dims)")
        return first(idxs)
    end

    # Some datasets combine flavor and nu/antinu into a single species
    # dimension of size 6. Detect that case and reshape directly.
    if (idx_flav === nothing || idx_nnu === nothing) && idx_comp !== nothing && idx_species !== nothing
        idx_sample = sample_idx((idx_comp, idx_species))
        perm = (idx_sample, idx_comp, idx_species)
        F4p = perm == (1, 2, 3) ? F4 : permutedims(F4, perm)

        N, ncomp, nspecies = size(F4p)
        @assert ncomp == 4 "post-permute: expected 4 spacetime components, got $ncomp"
        @assert nspecies == 6 "post-permute: expected 6 species, got $nspecies"

        F = Array{Float32}(undef, N, 6, 4)
        @inbounds for k in 1:N, a in 1:6, μ in 1:4
            F[k, a, μ] = Float32(F4p[k, μ, a])
        end

        return F
    end

    @assert idx_comp !== nothing "could not find dim of size 4 (spacetime components) in F4"
    @assert idx_nnu  !== nothing "could not find dim of size 2 (nu/antinu) in F4"
    @assert idx_flav !== nothing "could not find dim of size 3 (flavor) in F4; if your data already combines flavor, store it as a size-6 species dimension instead."

    idx_sample = sample_idx((idx_comp, idx_nnu, idx_flav))

    # Permute so dims become: (N, 4, 2, 3)
    perm = (idx_sample, idx_comp, idx_nnu, idx_flav)
    F4p = perm == (1, 2, 3, 4) ? F4 : permutedims(F4, perm)

    N, ncomp, nnu, nflav = size(F4p)
    @assert ncomp == 4 "post-permute: expected 4 spacetime components, got $ncomp"
    @assert nnu  == 2 "post-permute: expected 2 neutrino/antineutrino, got $nnu"
    #@assert nflav == 3 "post-permute: expected 3 flavors, got $nflav"

    # Now F4p[k, μ, inu, f] with μ in {1..4}, inu in {1..2}, f in {1..3}
    # Flatten (inu, f) -> species index a = 1..6
    F = Array{Float32}(undef, N, 6, 4)

    @inbounds for k in 1:N
        for inu in 1:2              # 1=nu, 2=antinu
            for f in 1:3            # 1=e, 2=mu, 3=tau
                a = (inu - 1) * 3 + f
                for μ in 1:4        # μ: 1=x,2=y,3=z,4=t
                    F[k, a, μ] = Float32(F4p[k, μ, inu, f])
                end
            end
        end
    end

    return F
end

# Map 6-species fluxes (x,y,z,t ordering) to the 4-species layout
# (t,x,y,z) × (νe, νebar, νx, νxbar) required by Box3D.
function add_species!(Fbox::AbstractMatrix, col::Int, Fspec)
    Fbox[1, col] += Fspec[4]  # t
    Fbox[2, col] += Fspec[1]  # x
    Fbox[3, col] += Fspec[2]  # y
    Fbox[4, col] += Fspec[3]  # z
end

function set_species!(F6::Array{Float32,3}, k::Int, a::Int, col::Int,
                      Fbox::AbstractMatrix, scale::Real=1.0)
    F6[k, a, 1] = Float32(scale * Fbox[2, col])  # x
    F6[k, a, 2] = Float32(scale * Fbox[3, col])  # y
    F6[k, a, 3] = Float32(scale * Fbox[4, col])  # z
    F6[k, a, 4] = Float32(scale * Fbox[1, col])  # t
end

# Apply Box3D to six-species fluxes by combining μ/τ into νx/νxbar,
# running the 4-species solver, then splitting back evenly into μ/τ.
function box3d_flux_on_six(F6::Array{<:Real,3}; mu_split::Real=0.5, lebedev_order::Int=13)
    N, nspecies, ncomp = size(F6)
    @assert nspecies == 6 "expected 6 species for Box3D mapping"
    @assert ncomp == 4 "expected 4 components (x,y,z,t)"

    μ_frac = clamp(mu_split, 0.0, 1.0) #make sure mu_split is valid
    τ_frac = 1.0 - μ_frac #calculate τ_frac from μ_frac

    F_box = Array{Float32}(undef, N, 6, 4)
    F_box4 = zeros(Float64, 4, 4)

    #map to 4 species, then apply box3D then map back to 6 species
    for k in 1:N
        fill!(F_box4, 0.0)#make sure everything is zeroes

        add_species!(F_box4, 1, @view F6[k, 1, :])  # νe
        add_species!(F_box4, 2, @view F6[k, 4, :])  # ν̄e
        add_species!(F_box4, 3, @view F6[k, 2, :])  # νμ
        add_species!(F_box4, 3, @view F6[k, 3, :])  # ντ
        add_species!(F_box4, 4, @view F6[k, 5, :])  # ν̄μ
        add_species!(F_box4, 4, @view F6[k, 6, :])  # ν̄τ

        F_box4_out = box3d_flux(F_box4; lebedev_order=lebedev_order)

        # Map back to 6 species, splitting νx/ν̄x evenly into μ/τ.
        set_species!(F_box, k, 1, 1, F_box4_out)
        set_species!(F_box, k, 4, 2, F_box4_out)
        set_species!(F_box, k, 2, 3, F_box4_out, μ_frac)
        set_species!(F_box, k, 3, 3, F_box4_out, τ_frac)
        set_species!(F_box, k, 5, 4, F_box4_out, μ_frac)
        set_species!(F_box, k, 6, 4, F_box4_out, τ_frac)
    end

    return F_box
end
