"""
Compute residual ΔF = F_true - F_box and flatten
6 species × 4 components -> (N, 24).
"""
function compute_residual(F_true::Array{<:Real,3}, F_box::Array{<:Real,3})
    @assert size(F_true) == size(F_box) "F_true and F_box must have same shape"
    N, nspecies, ncomp = size(F_true)
    @assert nspecies == 6 "expected 6 species"
    @assert ncomp == 4 "expected 4 components"

    residual = Array{Float32}(undef, N, nspecies * ncomp)

    @inbounds for k in 1:N
        idx = 1
        for a in 1:nspecies
            for μ in 1:ncomp
                residual[k, idx] = Float32(F_true[k, a, μ] - F_box[k, a, μ])
                idx += 1
            end
        end
    end

    return residual
end
