using Distributions

include("trees.jl")

norm_cdf(x) = cdf(Normal(0.0, 1.0), x)

# =============================================================================
# Binomial model
# =============================================================================

"""
    Binomial(steps, u, d)

Binomial tree pricing model.

Works for both European and American options, including dividend-paying underlyings.
Accuracy increases with `steps`; prices converge to the Black-Scholes-Merton value
as `steps → ∞`.

# Fields
- `steps::Int`: Number of time steps in the tree
- `u::Union{Float64, Nothing}`: factor greater than e^(r-δ)h, the up move of the spot price
- `d::Union{Float64, Nothing}`: factor less than e^(r-δ)h, the down move of the spot price

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)

price(EuropeanCall(40.0, 1.0), Binomial(100), data)              # CRR tree by default
price(AmericanPut(40.0, 1.0),  Binomial(100), data, ForwardTree())
```

See also: [`price`](@ref), [`get_full_tree`](@ref), [`BlackScholes`](@ref)
"""
struct Binomial
    steps::Int
    u::Union{Float64, Nothing}
    d::Union{Float64, Nothing}
end

Binomial(steps::Int) = Binomial(steps, nothing, nothing)

# =============================================================================
# Tree parameters (u, d, p*)
# =============================================================================

"""
    up_down_factors(tree::BinomialTree, data::MarketData, h) -> (u, d)

Default up and down factors for each tree type, used when the [`Binomial`](@ref)
model does not supply its own u and d.

- [`CRRTree`](@ref):       u = e^{σ√h},                d = e^{-σ√h}
- [`ForwardTree`](@ref):   u = e^{(r-δ)h + σ√h},       d = e^{(r-δ)h - σ√h}
- [`LognormalTree`](@ref): u = e^{(r-δ-σ²/2)h + σ√h},  d = e^{(r-δ-σ²/2)h - σ√h}

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
up_down_factors(ForwardTree(), data, 1.0 / 3)
```

See also: [`risk_neutral_probability`](@ref), [`extract_binomial_tree_parameters`](@ref)
"""
function up_down_factors end

function up_down_factors(::CRRTree, data::MarketData, h)
    (; vol) = data
    return exp(vol * sqrt(h)), exp(-vol * sqrt(h))
end

function up_down_factors(::ForwardTree, data::MarketData, h)
    (; rate, vol, div) = data
    drift = (rate - div) * h
    return exp(drift + vol * sqrt(h)), exp(drift - vol * sqrt(h))
end

function up_down_factors(::LognormalTree, data::MarketData, h)
    (; rate, vol, div) = data
    drift = (rate - div - vol^2 / 2) * h
    return exp(drift + vol * sqrt(h)), exp(drift - vol * sqrt(h))
end

"""
    risk_neutral_probability(tree::BinomialTree, data::MarketData, h) -> Float64

Closed-form risk-neutral probability p* for each tree type, valid for that
tree's default u and d (see [`up_down_factors`](@ref)):

- [`CRRTree`](@ref):       p* = (e^{(r-δ)h} - e^{-σ√h}) / (e^{σ√h} - e^{-σ√h})
- [`ForwardTree`](@ref):   p* = 1 / (1 + e^{σ√h})
- [`LognormalTree`](@ref): p* = (e^{σ²h/2} - e^{-σ√h}) / (e^{σ√h} - e^{-σ√h})

Each is the general formula p* = (e^{(r-δ)h} - d) / (u - d) simplified for that
tree's u and d. The forward and lognormal trees build the drift into u and d,
so r and δ cancel out and p* depends only on σ√h.

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
risk_neutral_probability(ForwardTree(), data, 1.0)  # ≈ 0.4256
```

See also: [`up_down_factors`](@ref), [`extract_binomial_tree_parameters`](@ref)
"""
function risk_neutral_probability(::CRRTree, data::MarketData, h)
    (; rate, vol, div) = data
    return (exp((rate - div) * h) - exp(-vol * sqrt(h))) /
           (exp(vol * sqrt(h)) - exp(-vol * sqrt(h)))
end

function risk_neutral_probability(::ForwardTree, data::MarketData, h)
    (; vol) = data
    return 1 / (1 + exp(vol * sqrt(h)))
end

function risk_neutral_probability(::LognormalTree, data::MarketData, h)
    (; vol) = data
    return (exp(vol^2 * h / 2) - exp(-vol * sqrt(h))) /
           (exp(vol * sqrt(h)) - exp(-vol * sqrt(h)))
end

"""
    risk_neutral_probability(u, d, data::MarketData, h) -> Float64

General risk-neutral probability for arbitrary up and down factors,
p* = (e^{(r-δ)h} - d) / (u - d). Used when the [`Binomial`](@ref) model supplies
its own u and d, since the tree-specific closed forms assume the default factors.
"""
function risk_neutral_probability(u, d, data::MarketData, h)
    (; rate, div) = data
    return (exp((rate - div) * h) - d) / (u - d)
end

"""
    extract_binomial_tree_parameters(tree::BinomialTree, model::Binomial, data::MarketData, h::Float64)

Calculate u, d and p* for a binomial tree.

If `model` has no u and d, the tree type's defaults from [`up_down_factors`](@ref)
and its closed-form p* from [`risk_neutral_probability`](@ref) are used. If `model`
supplies u and d, those are used with the general p* formula.

Errors if only one of u and d is given, or if p* ∉ [0, 1] (the tree admits arbitrage).

# Returns
A [`TreeParams`](@ref) with u, d, pu (p*), and pd (1-p*)

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
extract_binomial_tree_parameters(CRRTree(),       Binomial(100), data, 1.0 / 100)
extract_binomial_tree_parameters(ForwardTree(),   Binomial(100), data, 1.0 / 100)
extract_binomial_tree_parameters(LognormalTree(), Binomial(100), data, 1.0 / 100)
```

See also: [`BinomialTree`](@ref), [`TreeParams`](@ref)
"""
function extract_binomial_tree_parameters(tree::BinomialTree, model::Binomial, data::MarketData, h::Float64)
    if isnothing(model.u) && isnothing(model.d)
        u, d = up_down_factors(tree, data, h)
        pu = risk_neutral_probability(tree, data, h)
    elseif isnothing(model.u) || isnothing(model.d)
        error("Binomial model must supply both u and d, or neither")
    else
        u, d = model.u, model.d
        pu = risk_neutral_probability(u, d, data, h)
    end

    if pu < 0 || pu > 1
        error("No real arbitrage free p* (pu) exists")
    end

    return TreeParams(u, d, pu, 1 - pu)
end

# =============================================================================
# Full binomial tree: node and tree types
# =============================================================================

"""
    TreeNode

One node of a binomial tree, holding everything computed on the forward and
backward passes.

# Fields
- `step::Int`: time step `i` (0 = today, N = expiry)
- `ups::Int`: number of up moves `j` (0 ≤ j ≤ i)
- `time::Float64`: calendar time `i*h`
- `spot::Float64`: underlying price `S₀ uʲ dⁱ⁻ʲ`
- `payoff::Float64`: intrinsic (immediate exercise) value at this node
- `continuation::Float64`: discounted risk-neutral value of holding (NaN at expiry)
- `price::Float64`: option value at this node
- `delta::Float64`: shares Δ in the replicating portfolio (NaN at expiry)
- `bond::Float64`: bond holding B in the replicating portfolio (NaN at expiry)
- `early_exercise::Bool`: true if exercising here is optimal (always false for European options)
"""
Base.@kwdef mutable struct TreeNode
    step::Int
    ups::Int
    time::Float64
    spot::Float64
    payoff::Float64 = NaN
    continuation::Float64 = NaN
    price::Float64 = NaN
    delta::Float64 = NaN
    bond::Float64 = NaN
    early_exercise::Bool = false
end

"""
    FullBinomialTree

Triangular array of [`TreeNode`](@ref)s plus the [`TreeParams`](@ref) and step
size used to build it. Index with `t[i, j]`, where `i` is the time step and `j`
the number of up moves (both zero-based, matching textbook notation).
`t[0, 0]` is the root.
"""
struct FullBinomialTree
    nodes::Vector{Vector{TreeNode}}
    params::TreeParams
    h::Float64
end

Base.getindex(t::FullBinomialTree, i::Integer, j::Integer) = t.nodes[i + 1][j + 1]
nsteps(t::FullBinomialTree) = length(t.nodes) - 1
root(t::FullBinomialTree) = t[0, 0]

# =============================================================================
# Full binomial tree: shared building blocks
# =============================================================================

"""
    forward_pass(option, model::Binomial, data::MarketData, tree::BinomialTree)

Lay out every node with its step, up moves, time, spot and payoff, and set the
terminal prices to the payoff. Shared by all option types.

Returns `(nodes, params, h)`.
"""
function forward_pass(option, model::Binomial, data::MarketData, tree::BinomialTree)
    (; expiry) = option
    (; spot) = data
    N = model.steps
    h = expiry / N

    params = extract_binomial_tree_parameters(tree, model, data, h)
    (; u, d) = params

    nodes = [Vector{TreeNode}(undef, i + 1) for i in 0:N]
    for i in 0:N, j in 0:i
        S = spot * u^j * d^(i - j)
        nodes[i + 1][j + 1] = TreeNode(
            step = i, ups = j, time = i * h,
            spot = S, payoff = payoff(option, S),
        )
    end

    for n in nodes[N + 1]
        n.price = n.payoff
    end

    return nodes, params, h
end

"""
    replicate!(node, up, dn, params, disc, divdisc)

Fill in Δ, B and the continuation value of `node` from its two children.
Shared by all option types; only the choice of `node.price` differs.

    Δ = e^{-δh} (C_u - C_d) / (S(u - d))
    B = e^{-rh} (u C_d - d C_u) / (u - d)
    continuation = e^{-rh} (p* C_u + (1-p*) C_d) = Δ S + B
"""
function replicate!(node::TreeNode, up::TreeNode, dn::TreeNode, params, disc, divdisc)
    (; u, d, pu, pd) = params
    node.delta        = divdisc * (up.price - dn.price) / (up.spot - dn.spot)
    node.bond         = disc * (u * dn.price - d * up.price) / (u - d)
    node.continuation = disc * (pu * up.price + pd * dn.price)
    return node
end

# =============================================================================
# Full binomial tree: one method per exercise style
# =============================================================================

"""
    get_full_tree(option::EuropeanOption, model::Binomial, data::MarketData, tree::BinomialTree)

Build the full binomial tree for a European option. Every node holds its step,
up moves, time, spot, payoff, replicating portfolio (Δ, B), continuation value
and price. No early exercise check is performed, so each node's price equals
its continuation value and `early_exercise` is always `false`.

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
t = get_full_tree(EuropeanCall(40.0, 1.0), Binomial(3), data, CRRTree())
t[0, 0].price
t[2, 1].spot
```

See also: [`price`](@ref), [`delta`](@ref), [`B`](@ref)
"""
function get_full_tree(option::EuropeanOption, model::Binomial, data::MarketData, tree::BinomialTree)
    nodes, params, h = forward_pass(option, model, data, tree)
    disc, divdisc = exp(-data.rate * h), exp(-data.div * h)

    for i in (model.steps - 1):-1:0, j in 0:i
        n = nodes[i + 1][j + 1]
        replicate!(n, nodes[i + 2][j + 2], nodes[i + 2][j + 1], params, disc, divdisc)
        n.price = n.continuation
    end

    return FullBinomialTree(nodes, params, h)
end

"""
    get_full_tree(option::AmericanOption, model::Binomial, data::MarketData, tree::BinomialTree)

Build the full binomial tree for an American option. At each node the payoff
from exercising now is compared with the continuation value; the node's price
is the larger of the two and `early_exercise` records whether exercise won.
At expiry, `early_exercise` marks nodes where the option finishes in the money.

Δ and B at each node replicate the *continuation* value, so at nodes where
early exercise is optimal `Δ*S + B` equals `continuation`, not `price`.

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
t = get_full_tree(AmericanPut(40.0, 1.0), Binomial(3), data, CRRTree())
t[0, 0].price
[n.early_exercise for n in t.nodes[3]]
```

See also: [`price`](@ref), [`delta`](@ref), [`B`](@ref)
"""
function get_full_tree(option::AmericanOption, model::Binomial, data::MarketData, tree::BinomialTree)
    nodes, params, h = forward_pass(option, model, data, tree)
    disc, divdisc = exp(-data.rate * h), exp(-data.div * h)

    for n in nodes[end]
        n.early_exercise = n.payoff > 0
    end

    for i in (model.steps - 1):-1:0, j in 0:i
        n = nodes[i + 1][j + 1]
        replicate!(n, nodes[i + 2][j + 2], nodes[i + 2][j + 1], params, disc, divdisc)
        if n.payoff > n.continuation
            n.price = n.payoff
            n.early_exercise = true
        else
            n.price = n.continuation
        end
    end

    return FullBinomialTree(nodes, params, h)
end

# =============================================================================
# Binomial price, delta and B read off the tree
# =============================================================================

"""
    node_at(t::FullBinomialTree, step::Integer, ups::Integer) -> TreeNode

Return the node at time step `step` with `ups` up moves, checking that it
exists (0 ≤ step ≤ N and 0 ≤ ups ≤ step).
"""
function node_at(t::FullBinomialTree, step::Integer, ups::Integer)
    N = nsteps(t)
    0 <= step <= N || throw(ArgumentError("step must be between 0 and $N, got $step"))
    0 <= ups <= step || throw(ArgumentError("ups must be between 0 and step = $step, got $ups"))
    return t[step, ups]
end

"""
    interior_node_at(t::FullBinomialTree, step::Integer, ups::Integer, what::String) -> TreeNode

Like [`node_at`](@ref), but also rejects expiry nodes, where there is no
replicating portfolio (Δ and B are undefined).
"""
function interior_node_at(t::FullBinomialTree, step::Integer, ups::Integer, what::String)
    N = nsteps(t)
    step < N || throw(ArgumentError("$what is not defined at expiry (step $N); use step ≤ $(N - 1)"))
    return node_at(t, step, ups)
end

"""
    price(option, model::Binomial, data::MarketData, tree::BinomialTree = CRRTree(); step = 0, ups = 0)

Price a European or American option on a binomial tree, read off
[`get_full_tree`](@ref). Which `get_full_tree` method runs (with or without
early exercise) is chosen by the option's type.

By default this is the root (today's) price. Pass `step` and `ups` to get the
option value at any other node.

# Arguments
- `option`: a European option ([`EuropeanCall`](@ref), [`EuropeanPut`](@ref)) or an
  American option ([`AmericanCall`](@ref), [`AmericanPut`](@ref))
- `model::Binomial`: tree model (number of steps, and optional u and d)
- `data::MarketData`: market parameters (spot, rate, vol, div)
- `tree::BinomialTree`: [`CRRTree`](@ref) (default), [`ForwardTree`](@ref) or
  [`LognormalTree`](@ref), used to compute u, d (when not given in `model`) and p*

# Keywords
- `step::Integer = 0`: time step i, from 0 to `model.steps`
- `ups::Integer = 0`: number of up moves j, from 0 to `step`

# Returns
An [`AnalyticResult`](@ref) with the option price at that node.

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
price(EuropeanCall(40.0, 1.0), Binomial(3), data, CRRTree())        # root
price(AmericanPut(40.0, 1.0),  Binomial(3), data; step = 2, ups = 1)
```

See also: [`get_full_tree`](@ref), [`delta`](@ref), [`B`](@ref), [`BlackScholes`](@ref)
"""
function price(option, model::Binomial, data::MarketData, tree::BinomialTree;
               step::Integer = 0, ups::Integer = 0)
    t = get_full_tree(option, model, data, tree)
    return AnalyticResult(node_at(t, step, ups).price)
end

price(option, model::Binomial, data::MarketData; kwargs...) =
    price(option, model, data, CRRTree(); kwargs...)

"""
    delta(option, model::Binomial, data::MarketData, tree::BinomialTree = CRRTree(); step = 0, ups = 0) -> Float64

Binomial delta: the Δ of the replicating portfolio at a node of
[`get_full_tree`](@ref), Δ = e^{-δh} (C_u - C_d) / (S(u - d)).

By default this is the root. Pass `step` and `ups` for any node before expiry;
Δ is not defined at expiry nodes.

For a call, delta is in `[0, 1]`; for a put, in `[-1, 0]`.

# Keywords
- `step::Integer = 0`: time step i, from 0 to `model.steps - 1`
- `ups::Integer = 0`: number of up moves j, from 0 to `step`

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
delta(EuropeanCall(100.0, 1.0), Binomial(100), data, CRRTree())
delta(EuropeanCall(100.0, 1.0), Binomial(3), data; step = 1, ups = 1)
```

See also: [`price`](@ref), [`B`](@ref), [`get_full_tree`](@ref)
"""
function delta(option, model::Binomial, data::MarketData, tree::BinomialTree;
               step::Integer = 0, ups::Integer = 0)
    t = get_full_tree(option, model, data, tree)
    return interior_node_at(t, step, ups, "delta").delta
end

delta(option, model::Binomial, data::MarketData; kwargs...) =
    delta(option, model, data, CRRTree(); kwargs...)

"""
    B(option, model::Binomial, data::MarketData, tree::BinomialTree = CRRTree(); step = 0, ups = 0) -> Float64

Binomial bond position B of the replicating portfolio at a node of
[`get_full_tree`](@ref), B = e^{-rh} (u C_d - d C_u) / (u - d).

By default this is the root. Pass `step` and `ups` for any node before expiry;
B is not defined at expiry nodes.

B is negative for calls (borrowing) and positive for puts (lending).

# Keywords
- `step::Integer = 0`: time step i, from 0 to `model.steps - 1`
- `ups::Integer = 0`: number of up moves j, from 0 to `step`

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
B(EuropeanCall(100.0, 1.0), Binomial(100), data, CRRTree())
B(EuropeanPut(100.0, 1.0),  Binomial(3), data; step = 2, ups = 0)
```

See also: [`price`](@ref), [`delta`](@ref), [`get_full_tree`](@ref)
"""
function B(option, model::Binomial, data::MarketData, tree::BinomialTree;
           step::Integer = 0, ups::Integer = 0)
    t = get_full_tree(option, model, data, tree)
    return interior_node_at(t, step, ups, "B").bond
end

B(option, model::Binomial, data::MarketData; kwargs...) =
    B(option, model, data, CRRTree(); kwargs...)

# =============================================================================
# Full tree export and display helpers
# =============================================================================

"Flatten the tree into a vector of `Dict{Symbol,Any}`, one per node."
to_dicts(t::FullBinomialTree) =
    [Dict{Symbol,Any}(f => getfield(n, f) for f in fieldnames(TreeNode))
     for level in t.nodes for n in level]

"Flatten the tree into a vector of NamedTuples (works directly with `DataFrame(...)`)."
to_namedtuples(t::FullBinomialTree) =
    [NamedTuple{fieldnames(TreeNode)}(Tuple(getfield(n, f) for f in fieldnames(TreeNode)))
     for level in t.nodes for n in level]

"Print one field step by step; within each step nodes run j = 0 … i (down to up)."
function print_tree(t::FullBinomialTree, field::Symbol = :price; digits = 4)
    for level in t.nodes
        vals = (getfield(n, field) for n in level)
        println(join((v isa AbstractFloat ? round(v; digits) : v for v in vals), "  "))
    end
end

# =============================================================================
# Black-Scholes-Merton
# =============================================================================

"""
    BlackScholes()

Black-Scholes-Merton analytical pricing model for European options.

Provides exact closed-form solutions using the Black-Scholes-Merton formula. This is the
fastest and most accurate method for European options under the BSM assumptions
(constant volatility, log-normal price distribution, continuous dividend yield).

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
put = EuropeanPut(100.0, 1.0)

model = BlackScholes()
call_price = price(call, model, data)
put_price = price(put, model, data)
```

See also: [`price`](@ref), [`Binomial`](@ref)
"""
struct BlackScholes end

"""
    price(option::EuropeanCall, model::BlackScholes, data::MarketData)

Price a European call using the Black-Scholes-Merton closed-form formula.

# Returns
An [`AnalyticResult`](@ref) with the BSM call price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(EuropeanCall(100.0, 1.0), BlackScholes(), data)
```

See also: [`BlackScholes`](@ref), [`delta`](@ref), [`price`](@ref)
"""
function price(option::EuropeanCall, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    p = spot * exp(-div * expiry) * norm_cdf(d1) -
        strike * exp(-rate * expiry) * norm_cdf(d2)

    return AnalyticResult(p)
end

"""
    price(option::EuropeanPut, model::BlackScholes, data::MarketData)

Price a European put using the Black-Scholes-Merton closed-form formula.

# Returns
An [`AnalyticResult`](@ref) with the BSM put price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(EuropeanPut(100.0, 1.0), BlackScholes(), data)
```

See also: [`BlackScholes`](@ref), [`delta`](@ref), [`price`](@ref)
"""
function price(option::EuropeanPut, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    p = strike * exp(-rate * expiry) * norm_cdf(-d2) -
        spot * exp(-div * expiry) * norm_cdf(-d1)

    return AnalyticResult(p)
end

"""
    price(option::CashOrNothingCall, model::BlackScholes, data::MarketData)

Price a cash-or-nothing call using the Black-Scholes-Merton formula.

The analytical price is `cash * exp(-r*T) * N(d2)`.

# Returns
An [`AnalyticResult`](@ref) with the BSM binary call price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(CashOrNothingCall(100.0, 1.0, 1.0), BlackScholes(), data)
```

See also: [`CashOrNothingCall`](@ref), [`BlackScholes`](@ref)
"""
function price(option::CashOrNothingCall, model::BlackScholes, data::MarketData)
    (; strike, expiry, cash) = option
    (; spot, rate, vol, div) = data
    d2 = (log(spot / strike) + (rate - div - 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    return AnalyticResult(cash * exp(-rate * expiry) * norm_cdf(d2))
end

"""
    price(option::CashOrNothingPut, model::BlackScholes, data::MarketData)

Price a cash-or-nothing put using the Black-Scholes-Merton formula.

The analytical price is `cash * exp(-r*T) * N(-d2)`.

# Returns
An [`AnalyticResult`](@ref) with the BSM binary put price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(CashOrNothingPut(100.0, 1.0, 1.0), BlackScholes(), data)
```

See also: [`CashOrNothingPut`](@ref), [`BlackScholes`](@ref)
"""
function price(option::CashOrNothingPut, model::BlackScholes, data::MarketData)
    (; strike, expiry, cash) = option
    (; spot, rate, vol, div) = data
    d2 = (log(spot / strike) + (rate - div - 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    return AnalyticResult(cash * exp(-rate * expiry) * norm_cdf(-d2))
end

"""
    delta(option::EuropeanCall, model::BlackScholes, data::MarketData) -> Float64

Black-Scholes-Merton delta for a European call option.

Delta measures the sensitivity of the option price to a unit change in the spot price,
i.e., `∂V/∂S`. For a call, delta is in `[0, 1]`.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
delta(EuropeanCall(100.0, 1.0), BlackScholes(), data)  # ≈ 0.637
```

See also: [`price`](@ref), [`BlackScholes`](@ref)
"""
function delta(option::EuropeanCall, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    return exp(-div * expiry) * norm_cdf(d1)
end

"""
    delta(option::EuropeanPut, model::BlackScholes, data::MarketData) -> Float64

Black-Scholes-Merton delta for a European put option.

Delta measures the sensitivity of the option price to a unit change in the spot price,
i.e., `∂V/∂S`. For a put, delta is in `[-1, 0]`.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
delta(EuropeanPut(100.0, 1.0), BlackScholes(), data)  # ≈ -0.363
```

See also: [`price`](@ref), [`BlackScholes`](@ref)
"""
function delta(option::EuropeanPut, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    return -exp(-div * expiry) * norm_cdf(-d1)
end