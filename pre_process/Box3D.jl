# Box3D mixing scheme from Richers et al. (2025), Sec. III
# Implements Eqs. (6), (7), (20)–(22) to map initial number four-fluxes
# F_{α,s} (α = t,x,y,z; s = 1..4 species) to final fluxes after fast flavor conversion.
#
# Species ordering:
#   s=1 : ν_e
#   s=2 : ν̄_e
#   s=3 : ν_x  (combined μ/τ)
#   s=4 : ν̄_x (combined μ̄/τ̄)

using Lebedev

const FOUR_PI = 4π

# Total lepton number (νe + νx - ν̄e - ν̄x) using the number-density (t) component.
function lepton_number_4spec(F::AbstractMatrix)
    size(F, 1) >= 1 || error("F must have a t-component in row 1")
    size(F, 2) == 4 || error("F must be 4×4: (t,x,y,z) × (νe, νebar, νx, νxbar)")
    return Float64(F[1, 1] + F[1, 3] - F[1, 2] - F[1, 4])
end

# map flux factor f = |F|/N to Minerbo parameter Z via Eq. (7)
function z_from_flux_factor(f; tol=1e-10)
    f = clamp(f, 0.0, 1.0 - 1e-12)
    if f < 1e-6
        return 3.0 * f   # small-Z expansion
    end

    z_lo = 0.0
    z_hi = 50.0  # coth(50) - 1/50 ≈ 1

    for _ in 1:200
        z_mid = 0.5 * (z_lo + z_hi)
        z_mid = max(z_mid, 1e-12)
        sz = sinh(z_mid)
        cz = cosh(z_mid)
        r_mid = cz / sz - 1.0 / z_mid  # Eq. (7) RHS, i.e. coth(Z) - 1/Z
        if r_mid > f
            z_hi = z_mid
        else
            z_lo = z_mid
        end
        if z_hi - z_lo < tol
            return z_mid
        end
    end
    return 0.5 * (z_lo + z_hi)
end

# helper for Z/sinh(Z) with small-Z expansion
function shape_factor(Z)
    z = abs(Z)
    if z < 1e-4
        z2 = Z * Z
        return 1.0 - z2 / 6.0 + z2 * z2 / 120.0
    else
        return Z / sinh(Z)
    end
end

# build Minerbo-closure parameters (Z, F_hat, base) for one species
# so that f(n) = base * exp(Z * dot(F_hat, n)) reproduces (N, F)
function closure_params(N::Real, F::AbstractVector{<:Real})
    N ≤ 0 && return (0.0, (0.0, 0.0, 1.0), 0.0)

    Fx, Fy, Fz = F
    normF = sqrt(Fx * Fx + Fy * Fy + Fz * Fz)

    if normF ≤ 0
        # isotropic distribution
        Z = 0.0
        Fhat = (0.0, 0.0, 1.0)
        base = N / FOUR_PI
        return (Z, Fhat, base)
    end

    f = normF / N
    f = clamp(f, 0.0, 1.0 - 1e-12)
    Z = z_from_flux_factor(f)
    s = shape_factor(Z)
    base = N / FOUR_PI * s
    Fhat = (Fx / normF, Fy / normF, Fz / normF)
    return (Z, Fhat, base)
end

#Levedev grid instead of angular
function lebedev_grid(order::Int)
    @assert Lebedev.isavailable(order) "Lebedev rule of order $order not available"
    x, y, z, w_unit = Lebedev.lebedev_by_order(order)
    ndir = length(x)

    dirs = Vector{NTuple{3,Float64}}(undef, ndir)
    for i in 1:ndir
        dirs[i] = (x[i], y[i], z[i])
    end

    w = FOUR_PI .* w_unit
    return dirs, w
end

"""
    box3d_flux(F; nθ=32, nφ=64)

Apply the Box3D mixing scheme to an initial set of number four-fluxes.

Input:
  * F : 4×4 matrix of initial fluxes F(α,s), with
      α = 1: t component (number density),
          2: x, 3: y, 4: z components (spatial flux),
      s = 1: νₑ, 2: ν̄ₑ, 3: νₓ (μ/τ), 4: ν̄ₓ (μ̄/τ̄).

Keyword args:
  * nθ, nφ : angular resolution of the internal discretization.

Output:
  * F : 4×4 matrix of final fluxes in the same layout.
"""
function box3d_flux(F::AbstractMatrix{<:Real}; lebedev_order::Int=13)
    size(F, 1) == 4 || error("F must be 4×4: (t,x,y,z) × (νe, νebar, νx, νxbar)")
    size(F, 2) == 4 || error("F must be 4×4: (t,x,y,z) × (νe, νebar, νx, νxbar)")

    L_in = lepton_number_4spec(F)

    # unpack initial moments
    Ne, Fe = F[1, 1], F[2:4, 1]
    Nē, Fē = F[1, 2], F[2:4, 2]
    Nx, Fx = F[1, 3], F[2:4, 3]
    Nx̄, Fx̄ = F[1, 4], F[2:4, 4]

    # build Minerbo distributions for each species (Eq. 6–7)
    Ze, ê, base_e = closure_params(Ne, Fe)
    Zē, ê̄, base_ē = closure_params(Nē, Fē)
    Zx, x̂, base_x = closure_params(Nx, Fx)
    Zx̄, x̂̄, base_x̄ = closure_params(Nx̄, Fx̄)

    # Lebedev grid
    dirs, w = lebedev_grid(lebedev_order)
    ndir = length(dirs)

    fe = zeros(Float64, ndir)
    fē = zeros(Float64, ndir)
    fx = zeros(Float64, ndir)
    fx̄ = zeros(Float64, ndir)
    G = zeros(Float64, ndir)

    ex, ey, ez = ê
    ex̄, eȳ, ez̄ = ê̄
    xx, xy, xz = x̂
    xx̄, xȳ, xz̄ = x̂̄

    # build initial angular distributions and G(n)
    for k in 1:ndir
        nx, ny, nz = dirs[k]

        fe[k] = base_e * exp(Ze * (ex * nx + ey * ny + ez * nz))
        fē[k] = base_ē * exp(Zē * (ex̄ * nx + eȳ * ny + ez̄ * nz))
        fx[k] = base_x * exp(Zx * (xx * nx + xy * ny + xz * nz))
        fx̄[k] = base_x̄ * exp(Zx̄ * (xx̄ * nx + xȳ * ny + xz̄ * nz))

        G[k] = (fe[k] - fē[k]) - (fx[k] - fx̄[k])
    end

    # integrals of the positive / negative parts of G (Eq. 20)
    Iplus = 0.0
    Iminus = 0.0
    for k in 1:ndir
        g = G[k]
        wk = w[k]
        if g > 0
            Iplus += wk * g
        elseif g < 0
            Iminus += wk * (-g)  # store |g|
        end
    end

    # if no ELN-XLN crossing, no fast conversion
    if Iplus == 0.0 || Iminus == 0.0
        return copy(F)
    end

    # identify "small" and "large" ELN–XLN sides in |I| (Ω<, Ω>) and survival probs (Eq. 21)
    small_is_pos = Iplus ≤ Iminus
    Ismall = small_is_pos ? Iplus : Iminus
    Ilarge = small_is_pos ? Iminus : Iplus

    Psmall = 1.0 / 3.0
    Plarge = 1.0 - 2.0 * Ismall / (3.0 * Ilarge)
    Plarge = clamp(Plarge, 1.0 / 3.0, 1.0)

    P = similar(G)
    for k in 1:ndir
        g = G[k]
        if g == 0
            P[k] = Plarge  # measure-zero set; irrelevant how you treat it
        elseif (g > 0) == small_is_pos
            P[k] = Psmall
        else
            P[k] = Plarge
        end
    end

    # final angular distributions (Eq. 22)
    fe′ = similar(fe)
    fē′ = similar(fē)
    fx′ = similar(fx)
    fx̄′ = similar(fx̄)

    for k in 1:ndir
        p = P[k]
        fe_k, fē_k = fe[k], fē[k]
        fx_k, fx̄_k = fx[k], fx̄[k]

        fe′[k] = p * fe_k + (1 - p) * fx_k
        fē′[k] = p * fē_k + (1 - p) * fx̄_k
        fx′[k] = 0.5 * (1 - p) * fe_k + 0.5 * (1 + p) * fx_k
        fx̄′[k] = 0.5 * (1 - p) * fē_k + 0.5 * (1 + p) * fx̄_k
    end

    # integrate back to moments F′ (Eq. 1)
    Ne′ = 0.0
    Nē′ = 0.0
    Nx′ = 0.0
    Nx̄′ = 0.0
    Fe′ = zeros(3)
    Fē′ = zeros(3)
    Fx′ = zeros(3)
    Fx̄′ = zeros(3)

    for k in 1:ndir
        wk = w[k]
        nx, ny, nz = dirs[k]

        fev, fēv = fe′[k], fē′[k]
        fxv, fx̄v = fx′[k], fx̄′[k]

        Ne′ += wk * fev
        Nē′ += wk * fēv
        Nx′ += wk * fxv
        Nx̄′ += wk * fx̄v

        Fe′ .+= wk * fev .* [nx, ny, nz]
        Fē′ .+= wk * fēv .* [nx, ny, nz]
        Fx′ .+= wk * fxv .* [nx, ny, nz]
        Fx̄′ .+= wk * fx̄v .* [nx, ny, nz]
    end

    F′ = similar(F, 4, 4)
    F′[1, 1], F′[2:4, 1] = Ne′, Fe′
    F′[1, 2], F′[2:4, 2] = Nē′, Fē′
    F′[1, 3], F′[2:4, 3] = Nx′, Fx′
    F′[1, 4], F′[2:4, 4] = Nx̄′, Fx̄′

    return F′
end
