"""
    BinomialTree

Abstract type for tree selection in a Binomial model.

See also: [`BinomialModel`](@ref)
"""

abstract type BinomialTree end

"""
    ForwardTree <: BinomialTree

ForwardTree is McDonald's Convention
ForwardTree has the following up (u), down (d), and risk-neutral probability (p*) derivations

- u = exp((r-δ)*h + σ√h)
- d = exp((r-δ)*h - σ√h)
- p* = (1 - exp(-σ√h)) / (exp(σ√h) - exp(-σ√h))

"""
struct ForwardTree <: BinomialTree end

"""
    CRRTree <: BinomialTree

CRRTree is Cox-Ross-Rubinstein's Tree (Lattice)
CRRTree has the following up (u), down (d), and risk-neutral probability (p*) derivations

- u = exp(σ√h)
- d = exp(-σ√h)
- p* = (exp((r-δ)*h) - d) / (u - d)

"""
struct CRRTree <: BinomialTree end

"""
    LognormalTree <: BinomialTree

LognormalTree has the following up (u), down (d), and risk-neutral probability (p*) derivations

- u = exp((r-δ-σ^2/2)*h + σ√h)
- d = exp((r-δ-σ^2/2)*h - σ√h)
- p* = (exp((r-δ)*h) - d) / (u - d)

"""
struct LognormalTree <: BinomialTree end

"""
    TreeParams

TreeParams includes u, d, pu (p*), and pd (1-p*)
"""

struct TreeParams
    u::Float64
    d::Float64
    pu::Float64
    pd::Float64
end